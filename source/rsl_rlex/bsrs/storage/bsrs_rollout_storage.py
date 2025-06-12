# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.storage import RolloutStorage


class BsRsStorage(RolloutStorage):

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
        scale=0.1
    ):
        super().__init__(
            num_envs=num_envs,
            num_transitions_per_env = num_transitions_per_env,
            obs_shape = obs_shape,
            privileged_obs_shape = privileged_obs_shape,
            actions_shape = actions_shape,
            rnd_state_shape=rnd_state_shape,
            device="device")

        self.scale = scale

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        # 初始化 advantage 为 0，用于从尾部开始迭代计算
        advantage = 0
        # 从最后一个 transition 反向迭代至第 0 个 transition
        for step in reversed(range(self.num_transitions_per_env)):
            # 如果当前步骤是最后一步，则使用 bootstrap 值 last_values 作为下一状态值
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                # 否则，下一状态值为下一步存储的 V(s_{t+1})
                next_values = self.values[step + 1]
            # 判断下一状态是否为终止状态：如果终止，标记为 0；否则为 1
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # 计算 TD 误差：delta = r_t + gamma * V(s_{t+1}) * next_is_not_terminal - V(s_t)
            delta = self.rewards[step] + (next_is_not_terminal * gamma * next_values - self.values[step]) * self.scale
            # 递归计算优势值：
            # Advantage A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}) * next_is_not_terminal
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # 计算 return：R_t = V(s_t) + Advantage(s_t, a_t)
            self.returns[step] = advantage + self.values[step]

        # 计算每个时间步的 advantage： A = returns - values
        self.advantages = self.returns - self.values
        # 如果需要，标准化 advantage（避免再次标准化在 minibatch 中出现的问题）
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)