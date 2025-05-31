from isaaclab.utils import configclass
import math
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, TerminationTermCfg, ObservationTermCfg, \
                              ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclabex.envs.mdp.observations import privileged, ext_obs
from isaaclabex.envs.mdp.commands import commands_cfg


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands_cfg.ZeroSmallCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=commands_cfg.ZeroSmallCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi)
        ),
    )

    phase_command = commands_cfg.RefPoseWithPhaseCfg(
        resampling_time_range=(10.0, 10.0),
        period = 0.64,

        left_names = ['left_leg_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint'],
        right_names = ['right_leg_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint'],
        joint_scales = [0.17, 0.34, 0.17]
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        phase = ObservationTermCfg(func=ext_obs.phase_commands, params={"command_name": "phase_command"}, clip =(-18, 18))
        commands = ObservationTermCfg(func=mdp.generated_commands, scale = 0.25, params={"command_name": "base_velocity"}, clip = (-18, 18))
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.025, n_max=0.025), clip = (-18, 18))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale = 0.05, noise=Unoise(n_min=-0.25, n_max=0.25), clip = (-18, 18))

        actions = ObservationTermCfg(func=mdp.last_action, clip = (-18, 18))
        ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale = 1, noise=Unoise(n_min=-0.05, n_max=0.05), clip = (-18, 18))
        euler = ObservationTermCfg(func=ext_obs.root_euler_w, scale = 0.25, noise=Unoise(n_min=-0.015, n_max=0.015), clip = (-18, 18))

        history_length = 15

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        ## proprioceptive
        lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, scale = 2.0, noise=Unoise(n_min=-0.025, n_max=0.025), clip = (-18, 18))
        push_force = ObservationTermCfg(
            func=privileged.push_force,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        push_torque = ObservationTermCfg(
            func=privileged.push_torque,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        frictions = ObservationTermCfg(
            func=privileged.frictions,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )
        body_mass = ObservationTermCfg(
            func=privileged.body_mass, scale = 1 / 30,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }, clip = (-18, 18)
        )

        def __post_init__(self):
            self.history_length = 3

            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # startup

    startup_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.25),
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
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )

    interval_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
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
    height = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4})

    out_of_terrain = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
    contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
                            body_names=['base_link']), "threshold": 1.0},
    )
