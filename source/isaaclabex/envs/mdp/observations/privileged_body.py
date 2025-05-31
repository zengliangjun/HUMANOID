
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation, RigidObject

def body_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    return masses.to(env.device)

def push_force(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_vel_w[:, :2]

def push_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w

def frictions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    rigid_body_mass = ObsTerm(
            func=push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

    '''
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    material = asset.root_physx_view.get_material_properties()
    return material[:, :, 0]

