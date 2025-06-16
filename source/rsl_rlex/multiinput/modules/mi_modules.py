
from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation

def miencode_subclass(parent_class):

    class miencode(parent_class):

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

            policy_dims = 0
            for name in self.cfg["policy_groups"]:

                if name in self.cfg["encode_groups"]:
                    dim = cfg[f"encode_{name}_hidden_dims"][-1]
                else:
                    dim = cfg[f"obs_{name}_dim"]

                policy_dims += dim

            critic_dims = 0
            for name in self.cfg["critic_groups"]:

                if name in self.cfg["encode_groups"]:
                    dim = cfg[f"encode_{name}_hidden_dims"][-1]
                else:
                    dim = cfg[f"obs_{name}_dim"]

                critic_dims += dim

            cfg["num_actor_obs"] = policy_dims
            cfg["num_critic_obs"] = critic_dims

            super().__init__(**cfg)

            activation = resolve_nn_activation(cfg['activation'])

            for name in self.cfg["encode_groups"]:

                obs_dim = cfg[f"obs_{name}_dim"]
                hidden_dims = cfg[f"encode_{name}_hidden_dims"]
                module = self._build_encoder(activation, obs_dim, hidden_dims)
                setattr(self, f"encode_{name}_module", module)

            for name in self.cfg["encode_groups"]:
                module = getattr(self, f"encode_{name}_module")
                print(f"{name} MLP: {module}")


        def act_encode(self, observations):
            inputs = []
            for name in self.cfg["policy_groups"]:
                input = observations[name]
                if hasattr(self, f"encode_{name}_module"):
                    module = getattr(self, f"encode_{name}_module")
                    input = module(input)

                inputs.append(input)

            return torch.cat(inputs, dim = -1)

        def critic_encode(self, observations):
            inputs = []
            for name in self.cfg["critic_groups"]:
                input = observations[name]
                if hasattr(self, f"encode_{name}_module"):
                    module = getattr(self, f"encode_{name}_module")
                    input = module(input)

                inputs.append(input)

            return torch.cat(inputs, dim = -1)

        def act(self, observations, **kwargs):
            observations = self.act_encode(observations)
            return super().act(observations, **kwargs)

        def act_inference(self, observations):
            observations = self.act_encode(observations)
            return super().act_inference(observations)

        def evaluate(self, critic_observations, **kwargs):
            critic_observations = self.critic_encode(critic_observations)
            return super().evaluate(critic_observations, **kwargs)

    return miencode

from rsl_rl.modules import actor_critic, actor_critic_recurrent

MIEActorCritic = miencode_subclass(actor_critic.ActorCritic)
MIERecurrentActorCritic = miencode_subclass(actor_critic_recurrent.ActorCriticRecurrent)

