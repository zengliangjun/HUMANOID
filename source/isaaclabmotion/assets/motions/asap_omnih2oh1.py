
from __future__ import annotations
from typing import TYPE_CHECKING

import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

from isaaclabmotion.envs.managers.term_cfg import MotionsTermCfg
from isaaclabmotion.envs.motions.asap import ASAPMotions
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

OMNIH2OH1_CFG = MotionsTermCfg(
    func = ASAPMotions,
    resample_interval_s = 1000,
    assert_cfg = SceneEntityCfg("robot"),
    motion_file = f"{_ASSETS_ROOT}/motions/asap/omnih2o/stable_punch.pkl",
    # motion_file = f"{_ASSETS_ROOT}/motions/asap/omnih2o/amass_phc_filtered.pkl",
    params = {
        "mjcf_file": f"{_ASSETS_ROOT}/motions/asap/omnih2o/h1.xml",
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
            #{
            #    "joint_name": "head_link",
            #    "parent_name": "pelvis",
            #    "pos": [0.0, 0.0, 0.75],
            #    "rot": [1.0, 0.0, 0.0, 0.0]
            #}
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
              'left_hip_yaw_joint',
              'left_hip_roll_joint',
              'left_hip_pitch_joint',
              'left_knee_joint',
              'left_ankle_joint',

              'right_hip_yaw_joint',
              'right_hip_roll_joint',
              'right_hip_pitch_joint',
              'right_knee_joint',
              'right_ankle_joint',

              'torso_joint',

              'left_shoulder_pitch_joint',
              'left_shoulder_roll_joint',
              'left_shoulder_yaw_joint',
              'left_elbow_joint',
              'right_shoulder_pitch_joint',
              'right_shoulder_roll_joint',
              'right_shoulder_yaw_joint',
              'right_elbow_joint'
        ]
    }
)


