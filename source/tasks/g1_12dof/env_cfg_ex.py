from isaaclab.utils import configclass
from .mdps  import rewards_ex as rewards
from . env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

@configclass
class HKEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.HKSymmetryCfg()

@configclass
class HKEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.HKSymmetryCfg()

@configclass
class LREnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.LRSymmetryCfg()

@configclass
class LREnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.LRSymmetryCfg()


@configclass
class MMVMCfgEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.MMVMCfg()

@configclass
class MMVMCfgEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.MMVMCfg()


@configclass
class MMVM2CfgEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.MMVMCfg2()
        self.scene.robot.init_state.joint_pos["left_hip_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["left_knee_joint"] = 0
        self.scene.robot.init_state.joint_pos["left_ankle_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_hip_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_knee_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_ankle_pitch_joint"] = 0

@configclass
class MMVM2CfgEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = rewards.MMVMCfg2()
        self.scene.robot.init_state.joint_pos["left_hip_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["left_knee_joint"] = 0
        self.scene.robot.init_state.joint_pos["left_ankle_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_hip_pitch_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_knee_joint"] = 0
        self.scene.robot.init_state.joint_pos["right_ankle_pitch_joint"] = 0
