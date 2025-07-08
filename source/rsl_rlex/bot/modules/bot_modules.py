# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import importlib

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from . import bot, mappings

class BOTActorCritic(nn.Module):

    def _build_actor(self):
        matrix_str = self.cfg["shortest_path_matrix"]
        module_str, function_str = matrix_str.split(":")
        module = importlib.import_module(module_str)
        matrix = getattr(module, function_str)()


        actor_net = bot.BodyTransformer(
            matrix,
            self.cfg["embedding_dim"],
            self.cfg["feedforward_dim"],
            self.cfg["nheads"],
            self.cfg["nlayers"])

        mapping_str = self.cfg["actor_mapping"]
        module_str, mapping_str = mapping_str.split(":")
        module = importlib.import_module(module_str)
        mapping = getattr(module, mapping_str)

        self.cfg["actor_mapping"] = mapping

        self.actor = bot.BodyActor(
            mapping,
            actor_net,
            self.cfg["embedding_dim"],
            activation=resolve_nn_activation(self.cfg["activation"]))

        print(f"actor BOT: {self.actor}")

    def _build_critic(self):

        matrix_str = self.cfg["shortest_path_matrix"]
        module_str, function_str = matrix_str.split(":")
        module = importlib.import_module(module_str)
        matrix = getattr(module, function_str)()


        critic_net = bot.BodyTransformer(
            matrix,
            self.cfg["embedding_dim"],
            self.cfg["feedforward_dim"],
            self.cfg["nheads"],
            self.cfg["nlayers"])

        mapping_str = self.cfg["critic_mapping"]
        module_str, mapping_str = mapping_str.split(":")
        module = importlib.import_module(module_str)
        mapping = getattr(module, mapping_str)

        self.cfg["critic_mapping"] = mapping


        self.critic = bot.BodyCritic(
            mapping,
            critic_net,
            self.cfg["embedding_dim"])

        print(f"critic BOT: {self.critic}")

    """

    """
    is_recurrent = False

    def __init__(
        self,
        cfg
    ):
        super(BOTActorCritic, self).__init__()
        self.cfg = cfg
        self._build_actor()
        self._build_critic()

        mapping: mappings.token_mapping = self.cfg["actor_mapping"]

        num_actions = 0
        for id, name in enumerate(mapping.token_names):
                num_actions += mapping.output_dim(id)

        # Action noise
        self.std = nn.Parameter(self.cfg["init_noise_std"] * torch.ones(num_actions))
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    def setup(cfg, obs: dict):
        pass

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        std = self.std.expand_as(mean)
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
