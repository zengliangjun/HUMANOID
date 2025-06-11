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

@configclass
class G1FlatEntropyCfg(G1FlatCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.algorithm.class_name="EntropyPPO"
        self.algorithm.entropy_ranges = (4, 12)  # 目标熵值范围(最小,最大)  1.5, 10
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子
        self.experiment_name = "g1_12_entropy_flat"

@configclass
class G1FlatBSRSCfg(G1FlatCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.algorithm.class_name="BSRSPPO"
        self.algorithm.scale = 1.5
        self.experiment_name = "g1_12_bsrs15_flat"

@configclass
class G1FlatConstraintCfg(G1FlatCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.experiment_name = "g1_12_constraint_flat"
