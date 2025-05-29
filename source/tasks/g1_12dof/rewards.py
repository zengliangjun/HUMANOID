
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # -- task
    rew_alive = RewardTermCfg(
        func=reward_collect.is_alive,
        weight=0.15,
    )
    rew_tracking_linear = RewardTermCfg(
        func=reward_collect.track_lin_vel_xy_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_tracking_z = RewardTermCfg(
        func=reward_collect.track_ang_vel_z_exp,
        weight=.5,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_contact_with_phase = RewardTermCfg(
        func=reward_collect.rew_contact_with_phase,
        weight=0.18,
        params={"command_name": "phase_command",
                "sensor_cfg": SceneEntityCfg("contact_forces",
                     body_names=[".*left_ankle_roll_link", ".*right_ankle_pitch_link"])},
    )

    # base
    penalize_linear_z = RewardTermCfg(
        func=reward_collect.lin_vel_z_l2,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_linear_xy = RewardTermCfg(
        func=reward_collect.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_orientation = RewardTermCfg(
        func=reward_collect.flat_orientation_l2,
        weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_height = RewardTermCfg(
        func=reward_collect.base_height_l2,
        weight=-10.0, params={
            "target_height": 0.78,
            "asset_cfg": SceneEntityCfg("robot")}
    )

    # joint
    penalize_joint_acc = RewardTermCfg(
        func=reward_collect.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_joint_vel = RewardTermCfg(
        func=reward_collect.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_pos_limits = RewardTermCfg(
        func=reward_collect.joint_pos_limits,
        weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_torques = RewardTermCfg(
        func=reward_collect.joint_torques_l2,
        weight=-0.00001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_hip = RewardTermCfg(
        func=reward_collect.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg":
                SceneEntityCfg("robot",
                joint_names=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ])},
    )
    # feet
    penalize_foot_slide = RewardTermCfg(
        func=reward_collect.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    penalize_foot_clearance = RewardTermCfg(
        func=reward_collect.penalize_foot_clearance,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            'target_height': 0.08 + 0.055
        },
    )

    # action
    penalize_action_rate = RewardTermCfg(
        func=reward_collect.action_rate_l2, weight=-0.01)