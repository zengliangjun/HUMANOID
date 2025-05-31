
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation

def static_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    feet_contact = ObsTerm(
            func=static_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            },
        )
    '''
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    return materials[:, :, 0]

def dynamic_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    feet_contact = ObsTerm(
            func=dynamic_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            },
        )
    '''

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    materials = asset.root_physx_view.get_material_properties()
    return materials[:, :, 1]
