from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.multiinput.modules import mi_modules_cfg

experiment_name = "hover_hoverh1"

@configclass
class OMNIH2OH1CfgV0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000000
    save_interval = 500
    experiment_name = experiment_name
    empirical_normalization = False
    policy = mi_modules_cfg.MIEncodeActorCriticCfg(
        class_name = "MIEActorCritic",
        init_noise_std=1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        policy_groups= ["policy"],
        critic_groups= ["policy",
                        "critic"],
        encode_groups= [
            # "policy", "critic",
        ],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="MIPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
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
        #self.policy.rnn_type='lstm'
        #self.policy.rnn_hidden_size=512
        #self.policy.rnn_num_layers=1

        self.policy.encode_policy_hidden_dims = [768]
        self.policy.encode_critic_hidden_dims = [128]

        self.algorithm.entropy_ranges = (1.5, 30)  # 目标熵值范围(最小,最大)
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子


@configclass
class OMNIH2OH1CfgV1(OMNIH2OH1CfgV0):

    def __post_init__(self):
        super(OMNIH2OH1CfgV1, self).__post_init__()
        self.experiment_name = "hover_hoverh1_v1"

