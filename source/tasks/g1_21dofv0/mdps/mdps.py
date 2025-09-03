from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclabex.envs.mdp.statistics import joints
from isaaclabex.envs.managers import term_cfg

@configclass
class StatisticsCfg:
    pos = term_cfg.StatisticsTermCfg(
        func= joints.StatusJPos,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "step_joint_names": [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint",
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_elbow_joint", "right_elbow_joint",

                "left_hip_roll_joint",  "right_hip_roll_joint",
                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                "left_ankle_roll_joint","right_ankle_roll_joint",
                "left_ankle_pitch_joint","right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            ]},

        # episode_truncation = 80,
        export_interval = 1000000
    )
    action = term_cfg.StatisticsTermCfg(
        func= joints.StatusAction,
        params={
            "action_name": "joint_pos",
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "step_joint_names": [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "left_knee_joint", "right_knee_joint",
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_elbow_joint", "right_elbow_joint",

                "left_hip_roll_joint",  "right_hip_roll_joint",
                "left_hip_yaw_joint",   "right_hip_yaw_joint",
                "left_ankle_roll_joint","right_ankle_roll_joint",
                "left_ankle_pitch_joint","right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            ]},

        # episode_truncation = 80,
        export_interval = 1000000
    )

