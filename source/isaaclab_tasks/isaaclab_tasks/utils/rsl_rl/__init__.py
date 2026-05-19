# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RSL-RL architecture extensions.

Robot-agnostic model and algorithm classes that can be reused across any
IsaacLab locomotion task.  Import from here or from the sub-modules directly.
"""

from .algorithms.ppo_pfo import PPOWithPFO, RslRlPpoWithPfoCfg
from .models.linop import LinOpModel, RslRlLinOpModelCfg
from .models.simba import RslRlSimbaModelCfg, SimbaModel
