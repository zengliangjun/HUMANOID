from __future__ import annotations
from dataclasses import MISSING

from extends.motion_lib.aosp.motion_lib_base import MotionLibBase
from extends.motion_lib.aosp.torch_humanoid_batch import Humanoid_Batch
from isaaclab.utils import configclass

@configclass
class ASAPMotionlibCfg:
    """
    配置类，用于存储ASAP运动库的相关参数。
    """
    num_envs: int = MISSING
    device: str = MISSING
    ##
    step_dt: float = MISSING

    motion_file: str = MISSING

    mjcf_file: str = MISSING

    extend_config: list = MISSING


"""
- joint_name: "left_hand_link"
        parent_name: "left_elbow_link"
        pos: [0.3, 0.0, 0.0]
        rot: [1.0, 0.0, 0.0, 0.0]
- joint_name: "right_hand_link"
        parent_name: "right_elbow_link"
        pos: [0.3, 0.0, 0.0]
        rot: [1.0, 0.0, 0.0, 0.0]

"""


class MotionLibRobot(MotionLibBase):
    def __init__(self, config):

        super().__init__(config)
        self.mesh_parsers = Humanoid_Batch(config)
        self.output_device = config.device
        return

    def get_motion_length(self, motion_ids=None):
        if motion_ids is not None:
            motion_ids = motion_ids.cpu()
        _out = super(MotionLibRobot, self).get_motion_length(motion_ids)
        return _out.to(self.output_device)


    def sample_time(self, motion_ids, truncate_time=None):
        motion_ids = motion_ids.cpu()
        if truncate_time is not None:
            truncate_time = truncate_time.cpu()

        _out = super(MotionLibRobot, self).sample_time(motion_ids, truncate_time)
        return _out.to(self.output_device)

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_ids = motion_ids.cpu()
        motion_times = motion_times.cpu()
        if offset is not None:
            offset = offset.cpu()

        _out = super(MotionLibRobot, self).get_motion_state( motion_ids, motion_times, offset)
        for _key, _value in _out.items():
            _out[_key] = _value.to(self.output_device)

        return _out

    '''
    @property
    def joint_names(self):
        return self.mesh_parsers.joints_names

    @property
    def body_names(self):
        return self.mesh_parsers.body_names

    @property
    def augment_names(self):
        return self.mesh_parsers.body_names_augment
    '''
