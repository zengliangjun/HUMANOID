from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.multiinput.modules import mi_modules_cfg

@configclass
class G1ObsStatisticCfgV1(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 1000
    experiment_name = "g1obsStatisticv1" # "g1pbrsflat_noroll"  #
    empirical_normalization = False
    policy = mi_modules_cfg.MIEncodeActorCriticCfg(
        class_name = "MIERecurrentActorCritic",
        init_noise_std=1,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        policy_groups= ["policy", "action_statistics"],
        critic_groups= ["action_statistics",
                        "critic", "pos_statistics"],
        encode_groups= [
            "policy", "action_statistics",
            "critic", "pos_statistics"
        ],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MIPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
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
        self.policy.rnn_hidden_size=256
        self.policy.rnn_num_layers=1

        self.policy.encode_policy_hidden_dims = [96]
        self.policy.encode_action_statistics_hidden_dims = [72]

        self.policy.encode_critic_hidden_dims = [96]
        self.policy.encode_pos_statistics_hidden_dims = [72]

        self.algorithm.entropy_ranges = (1.5, 30)  # 目标熵值范围(最小,最大)
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子

