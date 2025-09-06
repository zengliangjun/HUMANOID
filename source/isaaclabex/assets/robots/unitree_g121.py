import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UNITREE_GO121_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_ROOT}/robots/usd/g121dof/g121dof.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8), #
        joint_pos={
            # lower body (12 dof)
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0,
            # waist (3 dof)
            "waist_yaw_joint": 0.0,
            #"waist_roll_joint": 0.0,
            #"waist_pitch_joint": 0.0,  # -90 degrees
            # upper body (14dof = 8 dof + 6 dof wrist)
            "left_shoulder_pitch_joint": 0.35, # 0.0,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint":  0.35,
            #"left_wrist_roll_joint":  0.0,
            #"left_wrist_pitch_joint":  0.0,
            #"left_wrist_yaw_joint":  0.0,
            "right_shoulder_pitch_joint": 0.35, # 0.0,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint":  0.35, # 0.35, #1, #0.87,
            #"right_wrist_roll_joint":  0.0,
            #"right_wrist_pitch_joint":  0.0,
            #"right_wrist_yaw_joint":  0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_pitch_joint": 0.0103,
                ".*_hip_roll_joint": 0.0251,
                ".*_hip_yaw_joint": 0.0103,
                ".*_knee_joint": 0.0251,
            },
        ),
        "feet": IdealPDActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2,
            armature=0.003597,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                #".*_wrist_roll_joint",
                #".*_wrist_pitch_joint",
                #".*_wrist_yaw_joint",
                ".*_elbow_joint",
                "waist_yaw_joint",
                #"waist_roll_joint",
                #"waist_pitch_joint",  # -90 degrees
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 90,
                ".*_shoulder_roll_joint": 60,
                ".*_shoulder_yaw_joint": 20,
                ".*_elbow_joint": 60,

                #".*_wrist_roll_joint": 4,
                #".*_wrist_pitch_joint": 4,
                #".*_wrist_yaw_joint": 4,

                "waist_yaw_joint": 400,
                #"waist_roll_joint": 400,
                #"waist_pitch_joint": 400,  # -90 degrees
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 1,
                ".*_shoulder_yaw_joint": .4,

                #".*_wrist_roll_joint": .2,
                #".*_wrist_pitch_joint": .2,
                #".*_wrist_yaw_joint": .2,

                ".*_elbow_joint": 1,
                "waist_yaw_joint": 5,
                #"waist_roll_joint": 5,
                #"waist_pitch_joint": 5,  # -90 degrees
            },
            armature={
                ".*_shoulder_pitch_joint": 0.003597,
                ".*_shoulder_roll_joint": 0.003597,
                ".*_shoulder_yaw_joint": 0.003597,

                #".*_wrist_roll_joint": 0.003597,
                #".*_wrist_pitch_joint": 0.003597,
                #".*_wrist_yaw_joint": 0.003597,

                ".*_elbow_joint": 0.003597,
                "waist_yaw_joint": 0.0103,
                #"waist_roll_joint": 0.0103,
                #"waist_pitch_joint": 0.0103,  # -90 degrees
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""
