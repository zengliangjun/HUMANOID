
from __future__ import annotations

import os
import statistics
import time

import torch
from collections import deque
import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

from rsl_rlex.multiinput.algorithms.mi_ppo import MIPPO
from rsl_rlex.multiinput.modules.mi_modules import MIEActorCritic, MIERecurrentActorCritic
from rsl_rlex.multiinput.modules.mi_modulesv2 import MIERecurrentActorCriticV2
from rsl_rlex.bot.modules.bot_modules import BOTActorCritic

from rsl_rl.utils import store_code_state

class MIPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg['algorithm']
        self.policy_cfg = train_cfg['policy']
        self.device = device
        self.env = env

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        obs = extras["observations"]

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # WMActorCritic
        actor_critic_class.setup(self.policy_cfg, obs)
        self.policy_cfg['num_actions'] = self.env.num_actions
        actor_critic = actor_critic_class(self.policy_cfg).to(self.device)

        # resolve dimension of rnd gated state
        if self.alg_cfg.get("rnd_cfg") is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for they key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if self.alg_cfg.get("symmetry_cfg") is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # init algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: MIPPO = alg_class(actor_critic, self.alg_cfg, device=self.device)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.empirical_normalization = False
        if self.empirical_normalization:
            self.obs_normalizer = None
            self.critic_obs_normalizer = None
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_actions],
            obs,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()

        obs = extras["observations"]
        for name, value in obs.items():
            obs[name] = value.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            for _ in range(self.num_steps_per_env):
                # Sample actions from policy

                with torch.inference_mode():
                    actions = self.alg.act(obs)
                # Step environment
                obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                with torch.inference_mode():
                    # Move to the agent device
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    obs = infos["observations"]
                    for name, value in obs.items():
                        obs[name] = value.to(self.device)

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, infos)

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                    stop = time.time()
                    collection_time = stop - start

                    # Learning step
                    start = stop
                    self.alg.compute_returns(obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
