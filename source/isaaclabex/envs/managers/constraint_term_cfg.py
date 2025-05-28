'''
**Constraint Manager**: [Constraints as Termination (CaT)](https://arxiv.org/abs/2403.18765) method implementation added.
'''

from __future__ import annotations
import torch

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Callable
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg


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
