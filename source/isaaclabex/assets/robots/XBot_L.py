import os.path as osp
_ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../../../assets"))

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

XBot_L_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_ROOT}/robots/usd/XBot-L/XBot.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95), #
        joint_pos={
            'left_leg_roll_joint': 0.,
            'left_leg_yaw_joint': 0.,
            'left_leg_pitch_joint': 0.,
            'left_knee_joint': 0.,
            'left_ankle_roll_joint': 0.,
            'left_ankle_pitch_joint': 0.,
            'right_leg_roll_joint': 0.,
            'right_leg_yaw_joint': 0.,
            'right_leg_pitch_joint': 0.,
            'right_knee_joint': 0.,
            'right_ankle_roll_joint': 0.,
            'right_ankle_pitch_joint': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
            'left_leg_roll_joint',
            'left_leg_pitch_joint',
            'left_leg_yaw_joint',
            'left_knee_joint',
            'right_leg_roll_joint',
            'right_leg_pitch_joint',
            'right_leg_yaw_joint',
            'right_knee_joint',
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_leg_roll_joint": 200.0,
                ".*_leg_pitch_joint": 350.0,
                ".*_leg_yaw_joint": 200.0,
                ".*_knee_joint": 350.0,
            },
            damping={
                ".*_leg_roll_joint": 10,
                ".*_leg_pitch_joint": 10,
                ".*_leg_yaw_joint": 10,
                ".*_knee_joint": 10,
            },
            #armature={
            #    ".*_leg_roll_joint": 0.0251,
            #    ".*_leg_pitch_joint": 0.0103,
            #    ".*_leg_yaw_joint": 0.0103,
            #    ".*_knee_joint": 0.0251,
            #},
        ),
        "feet": IdealPDActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            stiffness=15,
            damping=10,
            #armature=0.003597,
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""
