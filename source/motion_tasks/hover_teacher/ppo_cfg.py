from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.multiinput.modules import mi_modules_cfg

@configclass
class OMNIH2OH1CfgV0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000000
    save_interval = 1000
    experiment_name = "hover_omnih2oh1"
    empirical_normalization = False
    policy = mi_modules_cfg.MIEncodeActorCriticCfg(
        class_name = "MIERecurrentActorCritic",
        init_noise_std=1,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        policy_groups= ["policy"],
        critic_groups= ["policy",
                        "critic"],
        encode_groups= [
            "policy", "critic",
        ],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MIPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.2,
    )

    def __post_init__(self):
        self.policy.rnn_type='lstm'
        self.policy.rnn_hidden_size=256
        self.policy.rnn_num_layers=1

        self.policy.encode_policy_hidden_dims = [256]
        self.policy.encode_critic_hidden_dims = [256]

        self.algorithm.entropy_ranges = (1.5, 30)  # 目标熵值范围(最小,最大)
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子

