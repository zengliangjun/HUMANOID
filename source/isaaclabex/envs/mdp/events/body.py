from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import events
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_coms(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # 保存初始的中心质心坐标
        self.default_coms = self.asset.root_physx_view.get_coms()[..., :3].clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        coms_distribution_params: tuple[float, float],
        distribution: str = "uniform",
        operation: str = "add"
    ):
        # 如果未提供env_ids，则创建所有环境的索引
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # 根据asset_cfg解析身体索引，如果是全部则生成完整索引，否则转换成tensor
        if asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # 获取当前所有身体的重心坐标
        coms = self.asset.root_physx_view.get_coms()

        # 将指定环境和身体的重心坐标恢复为默认值
        coms[env_ids[:, None], body_ids, :3] = self.default_coms[env_ids[:, None], body_ids]

        # 对每个坐标维度执行随机化操作
        for id in range(3):
            coms[..., id] = events._randomize_prop_by_op(
                    coms[..., id], coms_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
                )

        # 更新物理视图中的重心坐标
        self.asset.root_physx_view.set_coms(coms, env_ids)

def push_by_setting_velocity_with_recovery_counters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    recovery_count: int = -1
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    if hasattr(env, "recovery_counters") and recovery_count != -1:
        reset = env.recovery_counters[env_ids].clone()
        reset[...] = recovery_count
        env.recovery_counters[env_ids] = torch.max(env.recovery_counters[env_ids], reset)

