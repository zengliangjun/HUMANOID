# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class XBotFlatCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 1000
    experiment_name = "xbot_l_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name = "ActorCritic",
        init_noise_std=1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.9,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XBotWithRefCfg(XBotFlatCfg):
    def __post_init__(self):
        self.experiment_name = "xbot_l_withref"