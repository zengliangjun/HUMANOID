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
