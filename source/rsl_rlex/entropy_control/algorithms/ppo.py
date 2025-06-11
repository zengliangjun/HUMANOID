from __future__ import annotations
import torch
from rsl_rl.algorithms import PPO
from torch import nn

class EntropyPPO(PPO):
    """带熵控制的PPO算法实现
    扩展标准PPO算法，增加动态熵系数调整功能
    用于平衡探索与利用
    """

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

        # 熵控制专用参数
        entropy_ranges = (1.5, 10),  # 目标熵值范围(最小,最大)
        entropy_coef_factor = 1.05,  # 熵系数调整幅度
        entropy_coef_scale = 10,  # 熵系数缩放因子
        entropy_smoothing = 0.9,  # 熵值平滑系数
        use_adaptive_ranges = True,  # 是否启用自适应熵范围
        reward_scale = 1.0  # 回报缩放因子
    ):
        super(EntropyPPO, self).__init__(
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

        self.entropy_ranges = entropy_ranges
        self.entropy_coef_factor = entropy_coef_factor
        self.entropy_coef_scale = entropy_coef_scale
        self.entropy_coef_org = entropy_coef

        self.entropy_smoothing = entropy_smoothing # 熵值平滑系数
        self.use_adaptive_ranges = use_adaptive_ranges # 是否启用自适应熵范围
        self.reward_scale = reward_scale # 回报缩放因子

    def update(self):  # noqa: C901
        """执行PPO算法更新步骤
        包含动态熵控制机制：
        1. 当平均熵低于目标范围时增大熵系数
        2. 当平均熵过高时减小熵系数
        3. 保持熵系数在合理范围内
        """
        mean_value_loss = 0      # 价值函数loss平均值
        mean_surrogate_loss = 0  # 策略loss平均值
        mean_entropy = 0         # 平均熵值
        # -- RND loss (探索奖励loss)
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss (对称性loss)
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], is_critic=False
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], is_critic=True
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 动态调整熵系数 - 优化版本
            current_entropy = entropy_batch.mean()

            # 平滑处理熵值
            if not hasattr(self, 'smoothed_entropy'):
                self.smoothed_entropy = current_entropy
            else:
                self.smoothed_entropy = self.entropy_smoothing * self.smoothed_entropy + \
                                       (1 - self.entropy_smoothing) * current_entropy

            # 自适应调整熵范围
            if self.use_adaptive_ranges:
                target_min = max(0.5, self.smoothed_entropy * 0.8)
                target_max = min(15.0, self.smoothed_entropy * 1.2)
                effective_ranges = (target_min, target_max)
            else:
                effective_ranges = self.entropy_ranges

            # 基于回报的调整因子
            if hasattr(self, 'last_reward'):
                reward_factor = 1.0 + (self.last_reward * self.reward_scale)
                reward_factor = torch.clamp(torch.tensor(reward_factor), 0.8, 1.2).item()
            else:
                reward_factor = 1.0

            # 动态调整熵系数
            if self.smoothed_entropy < effective_ranges[0]:  # 熵值过低
                adjust_factor = self.entropy_coef_factor * reward_factor
                self.entropy_coef = min(self.entropy_coef * adjust_factor,
                                      self.entropy_coef_scale * self.entropy_coef_org)

            elif self.smoothed_entropy > effective_ranges[1]:  # 熵值过高
                adjust_factor = (1.0 / self.entropy_coef_factor) * reward_factor
                self.entropy_coef = max(self.entropy_coef * adjust_factor,
                                      self.entropy_coef_org / self.entropy_coef_scale)


            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], is_critic=False
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], is_critic=False
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss
