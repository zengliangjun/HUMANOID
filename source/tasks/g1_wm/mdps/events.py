from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    startup_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (1, 1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # reset
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    interval_push = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    interval_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "gravity_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    interval_actuator = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            'stiffness_distribution_params': (.8, 1.2),
            'damping_distribution_params': (.8, 1.2),
            "operation": "scale",
        },
    )
    interval_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

