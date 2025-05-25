# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation
from rsl_rlex.wm import wm_cfg
from rsl_rl.modules import actor_critic_recurrent

class WMActorCritic(ActorCritic):
    is_recurrent = True
    cfg: wm_cfg.WMActorCriticCfg

    @staticmethod
    def setup(cfg, obs):
        if cfg['proprioceptive_names'] is not None:
            inputs = []
            for name in cfg['proprioceptive_names']:
                inputs.append(obs[name])
            inputs = torch.cat(inputs, dim=-1)
            if cfg['input_dim'] is not None:
                assert cfg['input_dim'] == inputs.shape[-1]
            else:
                cfg['input_dim'] = inputs.shape[-1]

        if cfg['continuous_names'] is not None:
            inputs = []
            for name in cfg['continuous_names']:
                inputs.append(obs[name])
            inputs = torch.cat(inputs, dim=-1)
            if cfg['continuous_dim'] is not None:
                assert cfg['continuous_dim'] == inputs.shape[-1]
            else:
                cfg['continuous_dim'] = inputs.shape[-1]

        if cfg['discreate_names'] is not None:
            inputs = []
            for name in cfg['discreate_names']:
                inputs.append(obs[name])
            inputs = torch.cat(inputs, dim=-1)
            if cfg['discreate_dim'] is not None:
                assert cfg['discreate_dim'] == inputs.shape[-1]
            else:
                cfg['discreate_dim'] = inputs.shape[-1]

    def _build_block(self, input_dim, dims, output_dim, activation):
        layers = []
        for layer_index, dim in enumerate(dims):
                layers.append(activation)
                layers.append(nn.Linear(input_dim, dim))
                input_dim = dim

        layers.append(activation)
        layers.append(nn.Linear(input_dim, output_dim))
        return layers

    def __init__(self, cfg: wm_cfg.WMActorCriticCfg):

        super().__init__(
            num_actor_obs=cfg['rnn_actor_hidden_size'],
            num_critic_obs=cfg['rnn_critic_hidden_size'],
            num_actions=cfg['num_actions'],
            actor_hidden_dims=cfg['actor_hidden_dims'],
            critic_hidden_dims=cfg['critic_hidden_dims'],
            activation=cfg['activation'],
            init_noise_std=cfg['init_noise_std'],
        )
        self.cfg = cfg
        activation = resolve_nn_activation(cfg['activation'])

        self.memory_encoder = actor_critic_recurrent.Memory(
                                    cfg['input_dim'],
                                    type=cfg['rnn_type'],
                                    num_layers=cfg['rnn_encoder_num_layers'],
                                    hidden_size=cfg['rnn_encoder_hidden_size'])


        self.memory_a = actor_critic_recurrent.Memory(
                            cfg['continuous_dim'] + cfg['discreate_dim'],
                            type=cfg['rnn_type'],
                            num_layers=cfg['rnn_actor_num_layers'],
                            hidden_size=cfg['rnn_actor_hidden_size'])
        self.a_activation = resolve_nn_activation(cfg['activation'])

        self.memory_c = actor_critic_recurrent.Memory(
                            cfg['continuous_dim'] + cfg['discreate_dim'],
                            type=cfg['rnn_type'],
                            num_layers=cfg['rnn_critic_num_layers'],
                            hidden_size=cfg['rnn_critic_hidden_size'])
        self.c_activation = resolve_nn_activation(cfg['activation'])


        print(f"Encoder RNN: {self.memory_encoder}")
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

        layers = self._build_block(
            cfg['rnn_encoder_hidden_size'],
            cfg['continuous_decoder_dims'],
            cfg['continuous_dim'],
            activation)

        self.continuous_decoder = nn.Sequential(*layers)

        layers = self._build_block(
            cfg['rnn_encoder_hidden_size'],
            cfg['discrete_decoder_dims'],
            cfg['discreate_dim'],
            activation)
        layers.append(nn.Sigmoid())

        self.discrete_decoder = nn.Sequential(*layers)

        print(f"Decoder continuous MLP: {self.continuous_decoder}")
        print(f"Decoder discrete MLP: {self.discrete_decoder}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)
        self.memory_encoder.reset(dones)

    def reconstruction(self, observations, masks=None, hidden_states=None):
        inputs = []
        for key in self.cfg['proprioceptive_names']:
            inputs.append(observations[key])
        inputs = torch.cat(inputs, dim=-1)
        encode = self.memory_encoder(inputs, masks, hidden_states).squeeze(0)
        continuous = self.continuous_decoder(encode)
        discrete = self.discrete_decoder(encode)
        return continuous, discrete

    def act_inference(self, observations):
        rc_observations = self.reconstruction(observations)
        rc_observations = torch.cat(rc_observations, dim = -1)
        input_a = self.memory_a(rc_observations).squeeze(0)
        input_a = self.a_activation(input_a)
        return super().act_inference(input_a)

    def act(self, observations, masks=None, hidden_states=None):
        if isinstance(observations, dict):
            rc_observations = self.reconstruction(observations)
            rc_observations = torch.cat(rc_observations, dim = -1)
        elif isinstance(observations, (list, tuple)):
            rc_observations = torch.cat(observations, dim = -1)
        elif isinstance(observations, torch.Tensor):
            rc_observations = observations
        else:
            raise f"can\'t support type{type(observations)}"

        input_a = self.memory_a(rc_observations, masks, hidden_states).squeeze(0)
        input_a = self.a_activation(input_a)
        return super().act(input_a)

    def evaluate(self, observations, masks=None, hidden_states=None):
        inputs = []
        for key in self.cfg['continuous_names']:
            inputs.append(observations[key])
        for key in self.cfg['discreate_names']:
            inputs.append(observations[key])
        inputs = torch.cat(inputs, dim=-1)
        input_c = self.memory_c(inputs, masks, hidden_states).squeeze(0)
        input_c = self.c_activation(input_c)
        return super().evaluate(input_c)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, \
               self.memory_c.hidden_states, \
               self.memory_encoder.hidden_states

    def continuous_observations(self, observations):
        inputs = []
        for key in self.cfg['continuous_names']:
            inputs.append(observations[key])
        inputs = torch.cat(inputs, dim=-1)
        return inputs

    def discrete_observations(self, observations):
        inputs = []
        for key in self.cfg['discreate_names']:
            inputs.append(observations[key])
        inputs = torch.cat(inputs, dim=-1)
        return inputs