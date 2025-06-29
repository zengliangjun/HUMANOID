from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence
from isaaclab.managers.manager_base import ManagerTermBase, ManagerBase
from isaaclab.assets import Articulation

import isaaclab.utils.math as math_utils
from extends.isaac_utils.rotations import xyzw_to_wxyz

from .term_cfg import MotionsTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class MotionsTerm(ManagerTermBase):

    cfg: MotionsTermCfg

    motion_ids: torch.Tensor
    motion_len: torch.Tensor
    start_times: torch.Tensor
    num_motions: int

    def __init__(self, cfg: MotionsTermCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        ## status
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

    @property
    def bodyAssetToMotionIds(self):
        return self._bodyAssetToMotionIds

    @property
    def jointMotionToAssetIds(self):
        return self._jointMotionToAssetIds

    """
    extend org info
    """
    def resolve_motion_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = True) -> tuple[list[int], list[str]]:
        import isaaclab.utils.string as string_utils
        return string_utils.resolve_matching_names(name_keys, self.motion_body_names, preserve_order)

    def resolve_extend_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = True) -> tuple[list[int], list[str]]:
        import isaaclab.utils.string as string_utils
        return string_utils.resolve_matching_names(name_keys, self.extend_body_names, preserve_order)

    @property
    def motion_body_names(self):
        return self._motion_body_names

    @property
    def extend_body_names(self):
        return self._extend_body_names

    @property
    def extend_body_parent_names(self):
        return self._extend_body_parent_names

    @property
    def extend_body_parent_ids(self):
        return self._extend_body_parent_ids

    @property
    def extend_body_parent_poses(self):
        return self._extend_body_parent_poses

    @property
    def extend_body_parent_rots_wxyz(self):
        return self._extend_body_parent_rots_wxyz

    @property
    def extend_body_parent_rots_xyzw(self):
        return self.extend_body_parent_rots_wxyz[:, :, [1, 2, 3, 0]]

    """
    extend org info
    """
    @property
    def extend_body_pos(self):
        return self._extend_body_pos

    @property
    def extend_body_rot_wxyz(self):  # sim
        return self._extend_body_rot_wxyz

    @property
    def extend_body_rot_xyzw(self):
        return self.extend_body_rot_wxyz[:, :, [1, 2, 3, 0]]

    @property
    def extend_body_ang_vel(self):
        return self._extend_body_ang_vel

    @property
    def extend_body_lin_vel(self):
        return self._extend_body_lin_vel

class MotionsManager(ManagerBase):

    _env: ManagerBasedEnv
    _motions: MotionsTerm = None
    _motions_name: str = None
    _motions_cfg: MotionsTermCfg = None
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        if self._motions_name is None:
            return "<MotionsManager> no active terms.\n"

        msg = f"<MotionsManager> contains {len(self._motions_name)} active terms.\n"
        msg += "\n"
        return msg

    """
    Properties.
    """
    @property
    def active_terms(self) -> list[str] | dict[str, list[str]]:
        return [self._motions_name]

    """
    Operations.
    """
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """

        if env_ids is None:
            env_ids = slice(None)

        if self._motions is not None:
            self._motions.reset(env_ids=env_ids)

        return {}

    def compute(self, isplay: bool = False) -> torch.Tensor:
        if self._motions is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        reset_time_outs = self._motions.termination_compute()
        self._motions.extend_compute()
        if isplay:
            self._motions.step_play()
        return reset_time_outs

    def get_term(self, name: str) -> MotionsTerm:
        assert name == self._motions_name
        return self._motions

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms

        assert len(cfg_items) == 1

        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, MotionsTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._motions = term_cfg.func
            self._motions_name = term_name
            self._motions_cfg = term_cfg

    def motion_ref(self, step: int):
        assert None != self._motions
        return self._motions.motion_ref(step)

    def motion_times(self, step: int):
        assert None != self._motions
        return self._motions.motion_times(step)
