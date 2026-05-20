# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner config for Anymal-D flat locomotion with LinOp factorized hidden layers."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.utils.rsl_rl.models.linop import RslRlLinOpModelCfg

from .rsl_rl_ppo_simba_cfg import AnymalDRoughPPORunnerCfg


@configclass
class AnymalDFlatPPOLinOpRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D flat locomotion — MLP-3 with LinOP factorized hidden layers."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_linop"
        # Orthogonal init (gain=2^(1/(2k))) gives condition number=1 for the
        # composed weight at init, eliminating the "balancing phase" that
        # random-Gaussian init caused (cond≈3e9 for k=8).  With a well-conditioned
        # start, 3e-4 is safe for all k; tighter grad norm guards later updates.
        self.algorithm.learning_rate = 3e-4
        self.algorithm.max_grad_norm = 0.5
        self.actor = RslRlLinOpModelCfg(
            hidden_dims=[512, 512, 512, 512, 256],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
            linop_num_factors=2,
        )
        self.critic = RslRlLinOpModelCfg(
            hidden_dims=[512, 512, 512, 512, 256],
            activation="elu",
            obs_normalization=False,
            linop_num_factors=2,
        )
