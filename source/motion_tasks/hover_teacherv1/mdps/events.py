import torch
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.events import body

@configclass
class EventCfg:

    # startup
    startup_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3), # (0.6, 1.25),
            "dynamic_friction_range": (1, 1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # interval
    interval_push = EventTermCfg(
        func=body.push_by_setting_velocity_with_recovery_counters,
        mode="interval",
        interval_range_s=(12.0, 18.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                "recovery_count": 60},
        is_global_time = True,
    )

    interval_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="interval",
        interval_range_s=(6, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
        is_global_time = True,
    )
    interval_coms = EventTermCfg(
        func=body.randomize_coms,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            #"coms_distribution_params": (torch.tensor([-0.1, -0.1, -0.1]), torch.tensor([0.1, 0.1, 0.1])),
            "coms_distribution_params": (-0.1, 0.1),
            "operation": "add",
            "distribution": "uniform",
        },
        is_global_time = True,
    )

    # reset
    reset_actuator = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            'stiffness_distribution_params': (0.75, 1.25),
            'damping_distribution_params': (0.75, 1.25),
            "operation": 'scale',
            "distribution": "uniform",
        },
    )
