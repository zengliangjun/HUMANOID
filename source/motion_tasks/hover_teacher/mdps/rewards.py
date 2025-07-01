from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclabex.envs.mdp.rewards import reward_collect
from isaaclabmotion.envs.mdps.rewards import motions_body, motions_joints, motions_foot

bnames= [
            'pelvis',

            'left_hip_yaw_link',
            'left_hip_roll_link',
            'left_hip_pitch_link',
            'left_knee_link',
            'left_ankle_link',

            'right_hip_yaw_link',
            'right_hip_roll_link',
            'right_hip_pitch_link',
            'right_knee_link',
            'right_ankle_link',

            'torso_link',

            'left_shoulder_pitch_link',
            'left_shoulder_roll_link',
            'left_shoulder_yaw_link',
            'left_elbow_link',
            'right_shoulder_pitch_link',
            'right_shoulder_roll_link',
            'right_shoulder_yaw_link',
            'right_elbow_link'
        ]

mname = "omnih2o"

ebnames = [
        "left_hand_link",
        "right_hand_link",
        "head_link"
    ]


@configclass
class RewardsCfg:
    # -- track
    # -- joints
    rew_track_jpos = RewardTermCfg(
        func=motions_joints.reward_track_joint_positions,
        weight=32,

        params={"sigma": 0.5,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot")},
    )

    rew_track_jvel = RewardTermCfg(
        func=motions_joints.reward_track_joint_velocities,
        weight=16,

        params={"sigma": 1,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot")},
    )

    # -- body_
    rew_track_blin = RewardTermCfg(
        func=motions_body.reward_track_body_lin_vel,
        weight=8,

        params={"sigma": 10,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot", body_names= bnames),
                "extend_body_names": ebnames },
    )

    rew_track_bang = RewardTermCfg(
        func=motions_body.reward_track_body_ang_vel,
        weight=8,

        params={"sigma": 10,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot", body_names= bnames),
                "extend_body_names": ebnames },
    )

    rew_track_lower_pos = RewardTermCfg(
        func=motions_body.reward_track_body_pos,
        weight=30.0 * 0.5,

        params={"sigma": 0.5,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot", body_names= bnames[:11]),
                "extend_body_names": [] },
    )
    rew_track_upper_pos = RewardTermCfg(
        func=motions_body.reward_track_body_pos,
        weight=30.0,

        params={"sigma": 0.03,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot", body_names= bnames[11:]),
                "extend_body_names": ebnames },
    )
    rew_track_vr_pos = RewardTermCfg(
        func=motions_body.reward_track_body_pos,
        weight=50.0,

        params={"sigma": 0.03,
                "motions_name": mname,
                "asset_cfg": SceneEntityCfg("robot", body_names= []),
                "extend_body_names": ebnames },
    )

    #
    p_torques = RewardTermCfg(
        func=reward_collect.penalize_torques_l2,
        weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_torque_limits = RewardTermCfg(
        func=reward_collect.penalize_torque_limits,
        weight=-2, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_jacc = RewardTermCfg(
        func=reward_collect.penalize_jacc_l2,
        weight=-0.000011, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_jvel = RewardTermCfg(
        func=reward_collect.penalize_jvel_l2,
        weight=-0.004, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_lower_actionrate = RewardTermCfg(
        func=reward_collect.penalize_action_rate2_l2,
        weight=-3.0, params={"asset_cfg":
            SceneEntityCfg("robot",
            joint_names=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_joint",
                "torso_joint",
            ])}
    )
    p_upper_actionrate = RewardTermCfg(
        func=reward_collect.penalize_action_rate2_l2,
        weight=-0.625, params={"asset_cfg":
            SceneEntityCfg("robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ])}
    )
    p_jpos_limits = RewardTermCfg(
        func=reward_collect.penalize_jpos_limits_l1,
        weight=-125.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_jvel_limits = RewardTermCfg(
        func=reward_collect.penalize_jvel_limits,
        weight=-50.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_jvel_limits = RewardTermCfg(
        func=reward_collect.penalize_jvel_limits,
        weight=-50.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    p_termination = RewardTermCfg(
        func=reward_collect.penalize_eps_terminated,
        weight=-250,
    )
    p_cforces = RewardTermCfg(
        func=reward_collect.penalize_forces,
        weight=-250,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names='*_ankle_link')}
    )
    p_stumble = RewardTermCfg(
        func=reward_collect.penalty_stumble,
        weight=-1000.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names='*_ankle_link')}
    )
    p_slippage = RewardTermCfg(
        func=reward_collect.penalize_slide_threshold,
        weight=-37.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names='*_ankle_link')}
    )
    p_feet_ori = RewardTermCfg(
        func=reward_collect.penalize_feet_orientation,
        weight=-37.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                body_names='*_ankle_link')}
    )
    rew_air_time = RewardTermCfg(
        func=motions_foot.reward_motion_feet_air_time,
        weight=1000.0,
        params={
                "motions_name": mname,
                "threshold": 0.25,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="*_ankle_link")},
    )
    p_both_air = RewardTermCfg(
        func=reward_collect.penalize_both_feet_in_air,
        weight=-200.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="*_ankle_link")},
    )
    p_feet_height = RewardTermCfg(
        func=reward_collect.penalize_max_feet_height_before_contact,
        weight=-2500.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="*_ankle_link")}
    )
