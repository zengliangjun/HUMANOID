
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
import rsl_rl.modules.actor_critic_recurrent as actor_critic_recurrent
from rsl_rl.modules import actor_critic


class MIERecurrentActorCriticV2(actor_critic.ActorCritic):
    is_recurrent = True

    @staticmethod
    def _build_encoder(activation, obs_dim, hidden_dims):
        layers = []

        input = obs_dim
        for id, output in enumerate(hidden_dims):

            layers.append(nn.Linear(input, output))

            if id != len(hidden_dims) - 1:
                layers.append(activation)

            input = output

        return nn.Sequential(*layers)

    @staticmethod
    def setup(cfg, obs: dict):
        for obs_name, ob in obs.items():
            cfg[f'obs_{obs_name}_dim'] = ob.shape[-1]

    cfg: dict
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        encode_policy_dims = 0
        for name in self.cfg["policy_groups"]:

            if name in self.cfg["encode_groups"]:
                dim = cfg[f"encode_{name}_hidden_dims"][-1]
            else:
                dim = cfg[f"obs_{name}_dim"]

            encode_policy_dims += dim

        encode_critic_dims = 0
        for name in self.cfg["critic_groups"]:

            if name in self.cfg["encode_groups"]:
                dim = cfg[f"encode_{name}_hidden_dims"][-1]
            else:
                dim = cfg[f"obs_{name}_dim"]

            encode_critic_dims += dim


        super().__init__(
            num_actor_obs=cfg["rnn_hidden_size"] + encode_policy_dims,
            num_critic_obs=cfg["rnn_hidden_size"] + encode_critic_dims,
            num_actions=cfg["num_actions"],
            actor_hidden_dims=cfg["actor_hidden_dims"],
            critic_hidden_dims=cfg["critic_hidden_dims"],
            activation=cfg['activation'],
            init_noise_std=cfg['init_noise_std'],
        )

        self.memory_a = actor_critic_recurrent.Memory(
            cfg["obs_policy_dim"],
            type=cfg["rnn_type"],
            num_layers=cfg["rnn_num_layers"],
            hidden_size=cfg["rnn_hidden_size"])

        self.memory_c = actor_critic_recurrent.Memory(
            cfg["obs_critic_dim"],
            type=cfg["rnn_type"],
            num_layers=cfg["rnn_num_layers"],
            hidden_size=cfg["rnn_hidden_size"])

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        activation = resolve_nn_activation(cfg['activation'])

        for name in self.cfg["encode_groups"]:

            obs_dim = cfg[f"obs_{name}_dim"]
            hidden_dims = cfg[f"encode_{name}_hidden_dims"]
            module = self._build_encoder(activation, obs_dim, hidden_dims)
            setattr(self, f"encode_{name}_module", module)

        for name in self.cfg["encode_groups"]:
            module = getattr(self, f"encode_{name}_module")
            print(f"{name} MLP: {module}")


    def act_encode(self, observations, masks=None, hidden_states=None):
        batch_mode = masks is not None
        inputs = []
        for name in self.cfg["policy_groups"]:
            input = observations[name]
            if batch_mode:
                input = actor_critic_recurrent.unpad_trajectories(input, masks)

            if hasattr(self, f"encode_{name}_module"):
                module = getattr(self, f"encode_{name}_module")
                input = module(input)

            inputs.append(input)

        return torch.cat(inputs, dim = -1)

    def critic_encode(self, observations, masks=None, hidden_states=None):
        batch_mode = masks is not None
        inputs = []
        for name in self.cfg["critic_groups"]:
            input = observations[name]
            if batch_mode:
                input = actor_critic_recurrent.unpad_trajectories(input, masks)

            if hasattr(self, f"encode_{name}_module"):
                module = getattr(self, f"encode_{name}_module")
                input = module(input)

            inputs.append(input)

        return torch.cat(inputs, dim = -1)

    def act(self, observations, **kwargs):
        encode = self.act_encode(observations, **kwargs)

        policy = observations["policy"]
        input_a = self.memory_a(policy, **kwargs)
        if len(kwargs) == 0:
            input_a = input_a.squeeze(0)
        observations = torch.cat([input_a.squeeze(0), encode], dim=-1)
        return super().act(observations, **kwargs)

    def act_inference(self, observations):
        encode = self.act_encode(observations)

        policy = observations["policy"]
        input_a = self.memory_a(policy)
        observations = torch.cat([input_a.squeeze(0), encode], dim=-1)
        return super().act_inference(observations)

    def evaluate(self, critic_observations, **kwargs):
        encode = self.critic_encode(critic_observations, **kwargs)

        critic = critic_observations["critic"]
        input_c = self.memory_c(critic, **kwargs)
        if len(kwargs) == 0:
            input_c = input_c.squeeze(0)

        critic_observations = torch.cat([input_c, encode], dim=-1)
        return super().evaluate(critic_observations, **kwargs)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

