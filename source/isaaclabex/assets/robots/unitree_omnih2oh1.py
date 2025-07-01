import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

OMNIH1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_ROOT}/robots/usd/omni_h1/omni_h1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.4, # -0.28,  # -16 degrees
            ".*_knee_joint": 0.8, # 0.79,  # 45 degrees
            ".*_ankle_joint": -0.4, # -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, # 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, # 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0, # 150.0,
                ".*_hip_roll_joint": 200.0, # 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0, # 200.0,
                "torso_joint": 300.0, # 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 6, # 5.0,
                "torso_joint": 6, # 5.0,
            },
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={".*_ankle_joint": 40.0}, # 20.0},
            damping={".*_ankle_joint": 2.0}, # 4.0},
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_shoulder_pitch_joint": 100, # 40.0,
                ".*_shoulder_roll_joint": 100, # 40.0,
                ".*_shoulder_yaw_joint": 100, # 40.0,
                ".*_elbow_joint": 100 # 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0, # 10.0,
                ".*_shoulder_roll_joint": 2.0, # 10.0,
                ".*_shoulder_yaw_joint": 2.0, # 10.0,
                ".*_elbow_joint": 2.0, # 10.0,
            },
        ),
    },
)
