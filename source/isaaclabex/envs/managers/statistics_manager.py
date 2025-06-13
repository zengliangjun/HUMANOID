from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any
from collections.abc import Sequence
from prettytable import PrettyTable
from isaaclab.managers.manager_base import ManagerTermBase, ManagerBase
from isaaclab.utils import configclass

from .term_cfg import StatisticsTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class StatisticsTerm(ManagerTermBase):
    """The base class for implementing a command term.

    A command term is used to generate commands for goal-conditioned tasks. For example,
    in the case of a goal-conditioned navigation task, the command term can be used to
    generate a target position for the robot to navigate to.

    It implements a resampling mechanism that allows the command to be resampled at a fixed
    frequency. The resampling frequency can be specified in the configuration object.
    Additionally, it is possible to assign a visualization function to the command term
    that can be used to visualize the command in the simulator.
    """

    def __init__(self, cfg: StatisticsTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # create buffers to store the command
        # -- metrics that can be used for logging
        self.statistics = dict()


class StatisticsManager(ManagerBase):

    _env: ManagerBasedRLEnv
    _terms: dict[str, StatisticsTerm] = dict()

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        msg = f"<StatisticsManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Statistics Terms"
        table.field_names = ["Index", "Name", "Type"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """
    @property
    def active_terms(self) -> list[str] | dict[str, list[str]]:
        return list(self._terms.keys())

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

        extras = {}
        for name, term in self._terms.items():
            metrics = term.reset(env_ids=env_ids)
            for metric_name, metric_value in metrics.items():
                extras[f"Statistics/{name}/{metric_name}"] = metric_value

        return extras

    def compute(self) -> None:
        for name, term in self._terms.items():
            term()

    def get_term(self, name: str) -> StatisticsTerm:
        return self._terms[name]

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, StatisticsTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._terms[term_name] = term_cfg.func
