
from __future__ import annotations
from typing import TYPE_CHECKING

import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

from isaaclabmotion.envs.managers.term_cfg import MotionsTermCfg
from isaaclabmotion.envs.motions.hover import HOVERMotions
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

HOVERH1_CFG = MotionsTermCfg(
    func = HOVERMotions,
    resample_interval_s = 1000,
    assert_cfg = SceneEntityCfg("robot"),
    # motion_file = f"{_ASSETS_ROOT}/motions/asap/omnih2o/stable_punch.pkl",
    motion_file = f"{_ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl",
    params = {
        "multi_thread": False,
        "mjcf_file": f"{_ASSETS_ROOT}/motions/asap/omnih2o/hover_h1.xml",
        "extend_config": [
            {
                "joint_name": "left_hand_link",
                "parent_name": "left_elbow_link",
                "pos": [0.3, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0]
            },
            {
                "joint_name": "right_hand_link",
                "parent_name": "right_elbow_link",
                "pos": [0.3, 0.0, 0.0],
                "rot": [1.0, 0.0, 0.0, 0.0]
            },
            {
                "joint_name": "head_link",
                "parent_name": "pelvis",
                "pos": [0.0, 0.0, 0.75],
                "rot": [1.0, 0.0, 0.0, 0.0]
            }
        ],
        "body_names": [
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
        ],
        "joint_names": [
              'left_hip_yaw',
              'left_hip_roll',
              'left_hip_pitch',
              'left_knee',
              'left_ankle',

              'right_hip_yaw',
              'right_hip_roll',
              'right_hip_pitch',
              'right_knee',
              'right_ankle',

              'torso',

              'left_shoulder_pitch',
              'left_shoulder_roll',
              'left_shoulder_yaw',
              'left_elbow',
              'right_shoulder_pitch',
              'right_shoulder_roll',
              'right_shoulder_yaw',
              'right_elbow'
        ]
    }
)


