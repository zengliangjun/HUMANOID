from __future__ import annotations
from rsl_rl.algorithms import PPO
from rsl_rlex.bsrs.storage.bsrs_rollout_storage import BsRsStorage

class BSRSPPO(PPO):

    def __init__(
        self,
        actor_critic,  # 策略-价值网络
        num_learning_epochs=1,  # 每个数据集的训练epoch数
        num_mini_batches=1,  # 小批量数量
        clip_param=0.2,  # PPO clip参数
        gamma=0.998,  # 折扣因子
        lam=0.95,  # GAE参数
        value_loss_coef=1.0,  # 价值函数loss系数
        entropy_coef=0.0,  # 初始熵系数
        learning_rate=1e-3,  # 学习率
        max_grad_norm=1.0,  # 梯度裁剪阈值
        use_clipped_value_loss=True,  # 是否裁剪价值loss
        schedule="fixed",  # 学习率调度方式
        desired_kl=0.01,  # 目标KL散度值
        device="cpu",  # 计算设备
        normalize_advantage_per_mini_batch=False,  # 是否标准化advantage
        # RND参数配置
        rnd_cfg: dict | None = None,
        # 对称性参数配置
        symmetry_cfg: dict | None = None,

        scale: float = 2,  # 缩放系数，用于调整奖励和价值函数
    ):
        super(BSRSPPO, self).__init__(
            actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            # RND parameters
            rnd_cfg = rnd_cfg,
            # Symmetry parameters
            symmetry_cfg = symmetry_cfg)

        self.scale = scale  # 缩放系数，用于调整奖励和价值函数

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = BsRsStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
            self.scale
        )
