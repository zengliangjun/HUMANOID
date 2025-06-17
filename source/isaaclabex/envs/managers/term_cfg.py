'''
**Constraint Manager**: [Constraints as Termination (CaT)](https://arxiv.org/abs/2403.18765) method implementation added.
'''

from __future__ import annotations
import torch

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Callable

from typing import TYPE_CHECKING, Any
from isaaclab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .statistics_manager import StatisticsTerm

@configclass
class ConstraintTermCfg(ManagerTermBaseCfg):
    """Configuration for a constraint term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the constraint signals as torch boolean tensors of
    shape (num_envs,).
    """

    time_out: int = 2
    """Whether the constraint term contributes towards episodic timeouts.
        ["constraint"]
         time_out
    Note:
        These usually correspond to tasks that have a fixed time limit.
    """

    gamma: float = 0.95
    """The discount factor."""

    probability_min: float = 0
    probability_max: float = 1.0
    """The maximum scaling factor for the termination probability for this constraint.

    For hard constraints, set p_max to 1.0 to strictly enforce the constraint.
    For soft constraints, use a value lower than 1.0 (e.g., 0.25) to allow some exploration,
    and optionally schedule p_max to increase over training.
    """
    """Whether to use a soft probability curriculum for this constraint.
    """

@configclass
class StatisticsTermCfg(ManagerTermBaseCfg):
    """Configuration for a manager term."""

    func: StatisticsTerm = MISSING
    """The function or class to be called for the term.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`ManagerTermBase` class
    and implement the required methods.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """
    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """

    export_interval: int = 1


    """
    统计过程中 episode 最大截断值, -1 表示不截断
    """
    episode_truncation: int = -1

