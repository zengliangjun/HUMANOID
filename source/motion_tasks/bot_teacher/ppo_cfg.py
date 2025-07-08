from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.bot.modules import bot_modules_cfg

experiment_name = "bot_hoverh1"

@configclass
class BOTH1CfgV0(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000000
    save_interval = 500
    experiment_name = experiment_name
    empirical_normalization = False
    policy = bot_modules_cfg.BOTActorCriticCfg(
        class_name = "BOTActorCritic",
        init_noise_std=1,

        activation="tanh",

        actor_mapping = "motion_tasks.bot_teacher.mdps.obs:actor_mapping",
        critic_mapping = "motion_tasks.bot_teacher.mdps.obs:critic_mapping",
        shortest_path_matrix = "isaaclabex.assets.utils.short_matrix_build:h1_sp_matrices",

        embedding_dim = 64,
        feedforward_dim = 128,

        nheads = 2,
        nlayers = 8
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

        self.algorithm.entropy_ranges = (1.5, 30)  # 目标熵值范围(最小,最大)
        self.algorithm.entropy_coef_factor = 1.05  # 熵系数调整幅度
        self.algorithm.entropy_coef_scale = 10  # 熵系数缩放因子
