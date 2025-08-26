from __future__ import annotations
from collections.abc import Sequence

import re
import os
import os.path as osp
import copy
from typing import TYPE_CHECKING
import torch
import torch.optim as optim

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from rsl_rlex.fld.modules import modules_cfg, fld_models
from .fldstatus import FldNormalization
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclabex.envs.managers.term_cfg import StatisticsTermCfg

class FLDCollect(ManagerTermBase):

    cfg: StatisticsTermCfg

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        params: dict = cfg.params
        # asset
        asset_cfg: SceneEntityCfg = params["asset_cfg"]
        self.asset: Articulation = self._env.scene[asset_cfg.name]
        # history
        self.record_status = {_id: [] for _id in range(self.num_envs)}
        self.history_status = []
        self.history_count = 0

        self._set_fld(params)


    def _set_fld(self, params):
        # fld
        assert "fld_module_cfg" in params
        fld_cfg : modules_cfg.FLDExtendCfg = params["fld_module_cfg"]
        self.fld_cfg = fld_cfg
        fld_cfg.step_dt = self._env.step_dt

        observation_dim = fld_cfg.observation_dim

        self.horizon = fld_cfg.observation_history_horizon + fld_cfg.forecast_horizon - 1

        ## module
        self.fld_module = fld_models.FLD(fld_cfg).to(self._env.device)
        self.status_normalizer = FldNormalization(shape = [observation_dim], until = 1.0e8).to(self._env.device)

        self.training = params["training"]

        self.fld_module.eval()
        self.status_normalizer.eval()

        if self.training:
            self.fld_training_module = copy.deepcopy(self.fld_module)
            self.fld_training_module.train()
            self.status_normalizer.train()

            self.fld_optimizer = optim.Adam(self.fld_training_module.parameters(), \
                                            lr = params["fld_learning_rate"], \
                                            weight_decay = params["fld_weight_decay"])

            assert hasattr(self._env.cfg, "log_dir")
            self.writer = SummaryWriter(log_dir = self._env.cfg.log_dir, flush_secs = 10)

            self.training_noise_level = params["training_noise_level"]
            self.fld_loss_scales = torch.tensor(params["fld_loss_scales"], dtype = torch.float32, device = self._env.device)[None, :]

            assert hasattr(self._env.cfg, "agent_cfg")
            self.agent_cfg = self._env.cfg.agent_cfg

            self.total_steps = 0
            self.total_iterations = 0
            self.total_loss = 0

        if hasattr(self._env.cfg, "tsp_checkpoint_path"):
            self._load_checkpoint(self._env.cfg.tsp_checkpoint_path)
        else:
            self._load(self._env.cfg.log_dir)


    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None or len(env_ids) == 0:
            return {}

        for id in env_ids:
            if isinstance(id, torch.Tensor):
                id = id.item()
            status = self.record_status[id]
            if 0 == len(status):
                continue
            if self.horizon * 1.5 > len(status):
                continue
            status = torch.vstack(status)

            status = status.unfold(0, self.horizon, 1).cpu()
            status = status.swapaxes(-2, -1)
            self.history_status.append(status)
            self.history_count += status.shape[0]

            self.record_status[id] = []

        if self.history_count > self.fld_cfg.mini_batch_size:
            self._train()
            self.history_status = []
            self.history_count = 0

        return {}

    def _get_ang_vel(self):
        return self.asset.data.root_ang_vel_b

    def _get_gravity(self):
        return self.asset.data.projected_gravity_b

    def _get_joint_pos(self):
        return self.asset.data.joint_pos - self.asset.data.default_joint_pos

    def _get_joint_vel(self):
        return self.asset.data.joint_vel - self.asset.data.default_joint_vel

    def _get_action(self):
        return self._env.action_manager.action

    def _get_commands(self, command_name: str = "base_velocity"):
        return self._env.command_manager.get_command(command_name)

    # LOAD SAVE
    def _save(self, path):
        file = osp.join(path, f"{self.fld_cfg.fldmodel_prefix}_{self.total_iterations}.pt")
        status = {
                "module_state_dict": self.fld_training_module.state_dict(),
                "normalizer_state_dict": self.status_normalizer.state_dict(),
                "optimizer_state_dict": self.fld_optimizer.state_dict(),
                "iterations": self.total_iterations,
            }

        torch.save(status, file)

        self.fld_module.load_state_dict(self.fld_training_module.state_dict())

    def _load(self, path):
        if not osp.exists(path):
            return

        models = [file for file in os.listdir(path) if re.match(f"{self.fld_cfg.fldmodel_prefix}_.*.pt", file)]
        if 0 == len(models):
            return

        models.sort(key=lambda m: "{0:0>20}".format(m))
        model = models[-1]
        model_path = osp.join(path, model)
        self._load_checkpoint(model_path)

    def _load_checkpoint(self, checkpoint_path):
        loaded_dict = torch.load(checkpoint_path)

        self.fld_module.load_state_dict(loaded_dict["module_state_dict"])
        self.status_normalizer.load_state_dict(loaded_dict["normalizer_state_dict"])
        if self.training:
            self.fld_training_module.load_state_dict(loaded_dict["module_state_dict"])
            self.fld_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.total_iterations = loaded_dict["iterations"]


    # TRAIN
    def _step_training(self, status):
        #  # self.fld_cfg.num_mini_batches x self.horizon x self.fld_cfg.observation_dim
        status = self.status_normalizer(status)

        # self.fld_cfg.num_mini_batches x self.fld_cfg.forecast_horizon x self.fld_cfg.observation_history_horizon x self.fld_cfg.observation_dim
        status = status.unfold(1, self.fld_cfg.observation_history_horizon, 1)
        status = status.swapaxes(-2, -1)
        inputs = status[:, 0]
        inputs_noised = inputs + torch.randn_like(inputs, device = self.device) * self.training_noise_level
        forecast_dynamics, latent, signal, params = self.fld_training_module.forward(inputs_noised, forecast_horizon = self.fld_cfg.forecast_horizon)

        loss = 0
        for i in range(self.fld_cfg.forecast_horizon):
            # compute loss for each step of forecast_horizon
            reconstruction_loss = self._compute_loss(forecast_dynamics[i, ...], status[:, i])
            loss += reconstruction_loss

        self.fld_optimizer.zero_grad()
        loss.backward()
        self.fld_optimizer.step()

        self.total_steps += 1
        self.total_loss += loss.item()

        if self.total_steps < self.agent_cfg.num_steps_per_env:
            return

        self.total_iterations += 1
        mean_fld_loss = self.total_loss / self.total_steps

        self.writer.add_scalar(f"fld_loss", mean_fld_loss, self.total_iterations)
        self.total_loss = 0
        self.total_steps = 0

        if self.total_iterations % self.agent_cfg.save_interval:
            return

        self._save(self._env.cfg.log_dir)

    def _train(self):
        status = torch.cat(self.history_status, dim = 0)

        batch = self.history_count // self.fld_cfg.num_mini_batches
        batch_size = self.fld_cfg.num_mini_batches * batch

        samples_indices = torch.randint(0, self.history_count, (batch_size * self.fld_cfg.num_epochs, ))
        for i in range(batch * self.fld_cfg.num_epochs):
            indices = samples_indices[i * self.fld_cfg.num_mini_batches: (i + 1) * self.fld_cfg.num_mini_batches]

            batch_status = status[indices]
            self._step_training(batch_status.to(self._env.device))


    def _compute_loss(self, input, target):
        input = self.status_normalizer.inverse(input)
        target = self.status_normalizer.inverse(target)

        diff = torch.square((input - target) * self.fld_loss_scales)
        return torch.mean(torch.sum(diff, dim=-1))

    def __call__(self):
        """执行统计计算"""
        status_list = []
        for status in self.cfg.params["status_names"]:
            status_fun = getattr(self, f"_get_{status}")
            status_list.append(status_fun())
        status_list = torch.cat(status_list, dim = -1)

        for _id in range(self.num_envs):
            self.record_status[_id].append(status_list[_id].cpu())

