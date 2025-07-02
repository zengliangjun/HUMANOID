# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from dataclasses import MISSING

import numpy as np
import torch
from dataclasses import dataclass

# TODO: Replace MotionLibH1 with the agnostic version when it's ready.
from .motion_lib_h1 import MotionLibH1
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree

@dataclass
class HOVERMotionlibCfg:
    num_envs: int = MISSING

    device: str = MISSING

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

- joint_name: "head_link",
        parent_name: "pelvis",
        pos: [0.0, 0.0, 0.75],
        rot: [1.0, 0.0, 0.0, 0.0]

"""

class MotionLibRobot():
    def __init__(
        self,
        cfg: HOVERMotionlibCfg,
    ):

        self.cfg = cfg
        self._motion_lib = MotionLibH1(cfg)
        self._skeleton_trees = [SkeletonTree.from_mjcf(cfg.mjcf_file)] * cfg.num_envs
        self.output_device = cfg.device

    def load_motions(self,
                     random_sample=True,
                     start_idx=0,
                     max_len=-1,
                     target_heading = None):

        """Loads motions from the motion dataset."""
        self._motion_lib.load_motions(
            skeleton_trees=self._skeleton_trees,
            gender_betas=[torch.zeros(17)] * self.cfg.num_envs,
            limb_weights=[np.zeros(10)] * self.cfg.num_envs,
            random_sample=random_sample,
            start_idx=start_idx,
            max_len = max_len,
            target_heading = target_heading,
        )

    def get_motion_state(self, motion_ids, motion_times, offset):
        if motion_ids is not None:
            motion_ids = motion_ids.cpu()
        if motion_times is not None:
            motion_times = motion_times.cpu()
        if offset is not None:
            offset = offset.cpu()

        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        for _key, _value in motion_res.items():
            motion_res[_key] = _value.to(self.output_device)
        return motion_res

    def get_motion_length(self, motion_ids=None):
        if motion_ids is not None:
            motion_ids = motion_ids.cpu()
        _out = self._motion_lib.get_motion_length(motion_ids)
        return _out.to(self.output_device)


    def sample_time(self, motion_ids, truncate_time=None):
        if motion_ids is not None:
            motion_ids = motion_ids.cpu()

        if truncate_time is not None:
            truncate_time = truncate_time.cpu()

        phase = torch.rand(motion_ids.shape)
        motion_len = self._motion_lib._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self.output_device)
