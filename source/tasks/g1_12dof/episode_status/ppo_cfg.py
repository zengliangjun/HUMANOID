from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class G1FlatCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 1000
    experiment_name = "g1episode" # "g1pbrsflat_noroll"  #
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name = "ActorCriticRecurrent",
        init_noise_std=0.8,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="EntropyPPO",
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
        self.policy.rnn_hidden_size=256
        self.policy.rnn_num_layers=1
        self.algorithm.entropy_ranges = (1.5, 10)  # 目标熵值范围(最小,最大)
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子



@configclass
class G1FlatCfgV2(G1FlatCfg):
    experiment_name = "g1episodev2" # "g1pbrsflat_noroll"  #

@configclass
class G1FlatCfgV3(G1FlatCfg):
    experiment_name = "g1episodev3" # "g1pbrsflat_noroll"  #



@configclass
class G1ObsStatisticCfgV0(G1FlatCfg):
    experiment_name = "g1obsStatisticv0" # "g1pbrsflat_noroll"  #
