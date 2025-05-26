from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, TerminationTermCfg, ObservationTermCfg, \
                              ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.observations import privileged


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        gravity = ObservationTermCfg(func=mdp.projected_gravity)
        commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)

        ## proprioceptive
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        contact_forces = ObservationTermCfg(
            func=privileged.feet_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
            }
        )
        payload = ObservationTermCfg(
            func=privileged.rigid_body_mass,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        stiffness = ObservationTermCfg(
            func=privileged.joint_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
        )

        damping = ObservationTermCfg(
            func=privileged.joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
        )
        # discrete
        contact_status = ObservationTermCfg(
            func=privileged.feet_contact_status,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    body_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    )
    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
