# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner configs for Anymal-D + SimBa + Proximal Feature Optimization (PFO).

Imports SimbaModel / RslRlSimbaModelCfg and RslRlPpoWithPfoCfg from the
shared utils package to avoid code duplication.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.utils.rsl_rl.algorithms.ppo_pfo import RslRlPpoWithPfoCfg
from isaaclab_tasks.utils.rsl_rl.models.simba import RslRlSimbaModelCfg

from .rsl_rl_ppo_cfg import AnymalDFlatPPORunnerCfg
from .rsl_rl_ppo_simba_cfg import AnymalDRoughPPORunnerCfg


@configclass
class AnymalDFlatPPOPfoSimbaRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D flat locomotion – SimBa actor/critic + PFO regularisation.

    Architecture is identical to AnymalDFlatPPOSimbaRunnerCfg; the only
    change is swapping the algorithm class to PPOWithPFO and adding the
    PFO-specific hyper-parameters.
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_simba_pfo"

        self.actor = RslRlSimbaModelCfg(
            hidden_dims=[128, 128, 128],
            activation="relu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
            simba_hidden_dim=128,
            simba_num_blocks=2,
            simba_final_gain=0.01,
            simba_depth_scale=True,
        )

        self.critic = RslRlSimbaModelCfg(
            hidden_dims=[128, 128, 128],
            activation="relu",
            obs_normalization=True,
            simba_hidden_dim=256,
            simba_num_blocks=2,
            simba_final_gain=1.0,
            simba_depth_scale=True,
        )

        self.algorithm = RslRlPpoWithPfoCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            pfo_coef=1.0,
            pfo_all_layers=False,
        )


@configclass
class AnymalDFlatPPOPfoRunnerCfg(AnymalDFlatPPORunnerCfg):
    """Anymal-D flat locomotion – vanilla MLP actor/critic + PFO regularisation.

    Identical architecture to AnymalDFlatPPORunnerCfg; only the algorithm is
    swapped to PPOWithPFO.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_flat_pfo"
        self.algorithm = RslRlPpoWithPfoCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            pfo_coef=1.0,
            pfo_all_layers=False,
        )
