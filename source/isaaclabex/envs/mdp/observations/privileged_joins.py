
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation

def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    joint_acc = ObsTerm(
            func=joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    joint_stiffness = ObsTerm(
            func=joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness

def joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    joint_stiffness = ObsTerm(
            func=joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping

def joint_torques(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    _torques = ObsTerm(
        func=joint_torques,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    '''
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque
