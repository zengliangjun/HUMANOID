
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

def feet_contact_status(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    '''
    feet_contact = ObsTerm(
            func=feet_contact_status,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )
    '''
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history  # N, T, B, 3
    forces = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    feet_contact = forces > threshold
    return feet_contact.float()


def feet_contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    feet_contact = ObsTerm(
            func=feet_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
        )
    '''
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return net_contact_forces.flatten(1)


def feet_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    '''
    feet_contact = ObsTerm(
            func=feet_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")
            },
        )
    '''

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w[:, None, :]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    _feet_pos = body_pos_w - root_pos_w

    return _feet_pos.flatten(1)
