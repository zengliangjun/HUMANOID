
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclabex.envs.mdp.rewards import reward_collect

@configclass
class RewardsCfg:
    # -- task
    rew_alive = RewardTermCfg(
        func=reward_collect.rewards_eps_alive,
        weight=0.15,
    )
    reward_lin_xy_exp = RewardTermCfg(
        func=reward_collect.reward_lin_xy_exp,
        weight=1.0,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_tracking_z = RewardTermCfg(
        func=reward_collect.reward_ang_z_exp,
        weight=0.5,
        params={"std": 0.25,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot")},
    )
    rew_contact_with_phase = RewardTermCfg(
        func=reward_collect.reward_contact_with_phase,
        weight=0.18,
        params={"command_name": "phase_command",
                "sensor_cfg": SceneEntityCfg("contact_forces",
                     body_names=[".*left_ankle_roll_link", ".*right_ankle_pitch_link"])},
    )

    # base
    penalize_linear_z = RewardTermCfg(
        func=reward_collect.penalize_lin_z_l2,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_linear_xy = RewardTermCfg(
        func=reward_collect.penalize_ang_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    penalize_orientation = RewardTermCfg(
        func=reward_collect.penalize_ori_l2,
        weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_height = RewardTermCfg(
        func=reward_collect.penalize_height_flat_or_rayl2,
        weight=-10.0, params={
            "target_height": 0.78,
            "asset_cfg": SceneEntityCfg("robot")}
    )

    # joint
    penalize_jacc_l2 = RewardTermCfg(
        func=reward_collect.penalize_jacc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_joint_vel = RewardTermCfg(
        func=reward_collect.penalize_jvel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_pos_limits = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_torques = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-0.00001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    penalize_hip = RewardTermCfg(
        func=reward_collect.penalize_jpos_deviation_l1,
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
        func=reward_collect.penalize_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    penalize_foot_clearance = RewardTermCfg(
        func=reward_collect.penalize_clearance,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            'target_height': 0.08 + 0.055
        },
    )

    # action
    penalize_action_rate = RewardTermCfg(
        func=reward_collect.penalize_action_rate_l2, weight=-0.01)