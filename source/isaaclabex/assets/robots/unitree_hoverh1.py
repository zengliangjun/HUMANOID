from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import H1_CFG

actuators = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
        effort_limit={
            ".*_hip_yaw": 200.0,
            ".*_hip_roll": 200.0,
            ".*_hip_pitch": 200.0,
            ".*_knee": 300.0,
            "torso": 200.0,
        },
        velocity_limit={
            ".*_hip_yaw": 23.0,
            ".*_hip_roll": 23.0,
            ".*_hip_pitch": 23.0,
            ".*_knee": 14.0,
            "torso": 23.0,
        },
        stiffness={
            ".*_hip_yaw": 150.0,
            ".*_hip_roll": 150.0,
            ".*_hip_pitch": 200.0,
            ".*_knee": 200.0,

            "torso": 200.0,
        },
        damping={
            ".*_hip_yaw": 5.0,
            ".*_hip_roll": 5.0,
            ".*_hip_pitch": 5.0,
            ".*_knee": 5.0,
            "torso": 5.0,
        },
    ),
    "feet": IdealPDActuatorCfg(
        joint_names_expr=[".*_ankle"],
        effort_limit=40,
        velocity_limit=9.0,
        stiffness=20.0,
        damping=4.0,
    ),
    "arms": IdealPDActuatorCfg(
        joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
        effort_limit={
            ".*_shoulder_pitch": 40.0,
            ".*_shoulder_roll": 40.0,
            ".*_shoulder_yaw": 18.0,
            ".*_elbow": 18.0,
        },
        velocity_limit={
            ".*_shoulder_pitch": 9.0,
            ".*_shoulder_roll": 9.0,
            ".*_shoulder_yaw": 20.0,
            ".*_elbow": 20.0,
        },
        stiffness= {
            ".*_shoulder_pitch": 40.0,
            ".*_shoulder_roll": 40.0,
            ".*_shoulder_yaw": 40.0,
            ".*_elbow": 40.0
        },
        damping={
            ".*_shoulder_pitch": 10.0,
            ".*_shoulder_roll": 10.0,
            ".*_shoulder_yaw": 10.0,
            ".*_elbow": 10.0
        },
    ),
}

H1HOVER_CFG: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

import torch

H1_SHORTPATH_MATRICES = torch.Tensor(
        [[0, 1, 1, 1,     2, 2, 2, 2,      3, 3, 3, 3,     4, 4, 4, 4,      5, 5, 5, 5],    # 0  ROOT
        [1, 0, 2,  2,     1, 3, 3, 3,      2, 4, 4, 4,     3, 5, 5, 5,      4, 6, 6, 6],     # 1  left_hip_yaw
        [1, 2, 0,  2,     3, 1, 3, 3,      4, 2, 4, 4,     5, 3, 5, 5,      6, 4, 6, 6],     # 2  right_hip_yaw_
        [1, 2, 2,  0,     3, 3, 1, 1,      4, 4, 2, 2,     5, 5, 3, 3,      6, 6, 4, 4],     # 3  torso

        [2, 1, 3,  3,     0, 4, 4, 4,      1, 5, 5, 5,     2, 6, 6, 6,      3, 7, 7, 7],     # 4  left_hip_roll
        [2, 3, 1,  3,     4, 0, 4, 4,      5, 1, 5, 5,     6, 2, 6, 6,      7, 3, 7, 7],     # 5  right_hip_roll
        [2, 3, 3,  1,     4, 4, 0, 2,      5, 5, 1, 3,     6, 6, 2, 4,      7, 7, 3, 5],     # 6  left_shoulder_pitch
        [2, 3, 3,  1,     4, 4, 2, 0,      5, 5, 3, 1,     6, 6, 4, 2,      7, 7, 5, 3],     # 7  right_shoulder_pitch

        [3, 2, 4,  4,     1, 5, 5, 5,      0, 6, 6, 6,     1, 7, 7, 7,      2, 8, 8, 8],     # 8  left_hip_pitch
        [3, 4, 2,  4,     5, 1, 5, 5,      6, 0, 6, 6,     7, 1, 7, 7,      8, 2, 8, 8],     # 9  right_hip_pitch
        [3, 4, 4,  2,     5, 5, 1, 3,      6, 6, 0, 4,     7, 7, 1, 5,      8, 8, 2, 6],     # 10  left_shoulder_roll
        [3, 4, 4,  2,     5, 5, 3, 1,      6, 6, 4, 0,     7, 7, 5, 1,      8, 8, 6, 2],     # 11  right_shoulder_roll

        [4, 3, 5,  5,     2, 6, 6, 6,      1, 7, 7, 7,     0, 8, 8, 8,      1, 9, 9, 9],     # 12  left_knee
        [4, 5, 3,  5,     6, 2, 6, 6,      7, 1, 7, 7,     8, 0, 8, 8,      9, 1, 9, 9],     # 13  right_knee
        [4, 5, 5,  3,     6, 6, 2, 4,      7, 7, 1, 5,     8, 8, 0, 6,      9, 9, 1, 7],     # 14  left_shoulder_yaw
        [4, 5, 5,  3,     6, 6, 4, 2,      7, 7, 5, 1,     8, 8, 6, 0,      9, 9, 7, 1],     # 15  right_shoulder_yaw

        [5, 4, 6,  6,     3, 7, 7, 7,      2, 8, 8, 8,     1, 9, 9, 9,      0,10,10,10],     # 16  left_ankle
        [5, 6, 4,  6,     7, 3, 7, 7,      8, 2, 8, 8,     9, 1, 9, 9,     10, 0,10,10],     # 17  right_ankle
        [5, 6, 6,  4,     7, 7, 3, 5,      8, 8, 2, 6,     9, 9, 1, 7,     10,10, 0, 8],     # 18  left_elbow
        [5, 6, 6,  4,     7, 7, 5, 3,      8, 8, 6, 2,     9, 9, 7, 1,     10,10, 8, 0]]     # 19  right_elbow
)

H1_hierarchy = {
        'root': None,
        'left_hip_yaw': 'root',
        'right_hip_yaw': 'root',
        'torso': 'root',

        'left_hip_roll': 'left_hip_yaw',
        'right_hip_roll': 'right_hip_yaw',
        'left_shoulder_pitch': 'torso',
        'right_shoulder_pitch': 'torso',

        'left_hip_pitch': 'left_hip_roll',
        'right_hip_pitch': 'right_hip_roll',
        'left_shoulder_roll': 'left_shoulder_pitch',
        'right_shoulder_roll': 'right_shoulder_pitch',

        'left_knee': 'left_hip_pitch',
        'right_knee': 'right_hip_pitch',
        'left_shoulder_yaw': 'left_shoulder_roll',
        'right_shoulder_yaw': 'right_shoulder_roll',

        'left_ankle': 'left_knee',
        'right_ankle': 'right_knee',
        'left_elbow': 'left_shoulder_yaw',
        'right_elbow': 'right_shoulder_yaw',
    }
