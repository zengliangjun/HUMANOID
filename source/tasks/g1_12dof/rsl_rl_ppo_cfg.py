# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class G1FlatCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 1000
    experiment_name = "g1_12_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name = "ActorCriticRecurrent",
        init_noise_std=0.8,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        self.policy.rnn_type='lstm'
        self.policy.rnn_hidden_size=64
        self.policy.rnn_num_layers=1

@configclass
class G1FlatExCfg(G1FlatCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.experiment_name = "g1_12_ex_flat"
