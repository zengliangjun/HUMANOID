
from __future__ import annotations

import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

from isaaclabmotion.envs.managers.term_cfg import MotionsTermCfg
from isaaclabmotion.envs.motions.asap import ASAPMotions
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

ASAPG129DOF23_CFG = MotionsTermCfg(
    func = ASAPMotions,

    assert_cfg = SceneEntityCfg("robot"),
    motion_file = f"{_ASSETS_ROOT}/motions/asap/asapg1/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl",
    params = {
        "mjcf_file": f"{_ASSETS_ROOT}/motions/asap/asapg1/g1_29dof_anneal_23dof_fitmotionONLY.xml",
        "extend_config": [
            {
                "joint_name": "left_hand_link",
                "parent_name": "left_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0]
            },
            {
                "joint_name": "right_hand_link",
                "parent_name": "right_elbow_link",
                "pos": [0.25, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0]
            },
            {
                "joint_name": "head_link",
                "parent_name": "torso_link",
                "pos": [0.0, 0.0, 0.42],
                "rot": [1.0, 0.0, 0.0, 0.0]
            }
        ],
        "body_names": [
            'pelvis',
            'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
            'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
            'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
            'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
            'waist_yaw_link', 'waist_roll_link', 'torso_link',
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
        ],
        "joint_names": [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
        ]
    }
)

