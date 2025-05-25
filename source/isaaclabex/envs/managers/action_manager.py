# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import AssetBase
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.managers import  action_manager

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ActionManagerExts(ManagerBase):

    _org : action_manager.ActionManager

    def __init__(self, action_org: action_manager.ActionManager):
        self._org = action_org

    def __str__(self) -> str:
        return self._org.__str__()

    """
    Properties.
    """
    @property
    def total_action_dim(self) -> int:
        return self._org.total_action_dim()

    @property
    def active_terms(self) -> list[str]:
        return self._org.active_terms

    @property
    def action_term_dim(self) -> list[int]:
        return self._org.action_term_dim

    @property
    def action(self) -> torch.Tensor:
        return self._org.action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._org.prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        return self._org.has_debug_vis_implementation

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return self._org.get_active_iterable_terms()

    def set_debug_vis(self, debug_vis: bool):
        self._org.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        return self._org.reset(env_ids)

    def process_action(self, action: torch.Tensor):
        self._org.process_action(action)

    def apply_action(self) -> None:
        self._org.apply_action()

    def get_term(self, name: str) -> action_manager.ActionTerm:
        return self._org.get_term(name)

class ActionManagerExts2(action_manager.ActionManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super(ActionManagerExts2, self).__init__(cfg, env)
