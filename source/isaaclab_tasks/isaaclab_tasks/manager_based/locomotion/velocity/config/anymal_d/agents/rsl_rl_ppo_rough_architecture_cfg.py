# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture variants for Anymal-D rough (perceptive) locomotion.

Mirrors the Flat architecture cfgs (rsl_rl_ppo_simba_cfg, rsl_rl_ppo_linop_cfg,
rsl_rl_ppo_pfo_simba_cfg) but pinned to the Rough training budget
(max_iterations=1500) and rough-specific experiment names. The base
AnymalDRoughPPORunnerCfg is already modern-style (actor/critic + obs_groups),
so each variant inherits from it cleanly and only swaps the model/algorithm.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.utils.rsl_rl.algorithms.ppo_pfo import RslRlPpoWithPfoCfg
from isaaclab_tasks.utils.rsl_rl.models.linop import RslRlLinOpModelCfg
from isaaclab_tasks.utils.rsl_rl.models.simba import RslRlSimbaModelCfg

from .rsl_rl_ppo_cfg import AnymalDRoughPPORunnerCfg


@configclass
class AnymalDRoughPPOSimbaRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D rough locomotion — SimBa actor/critic."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_rough_simba"
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


@configclass
class AnymalDRoughPPOLinOpRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D rough locomotion — MLP-3 with LinOp factorized hidden layers."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_rough_linop"
        # Same LR/grad-norm regime as flat LinOp: factored linear stack is
        # sensitive to Adam's first-step stride; orthogonal init via init_std=1.0
        # keeps condition number=1 at init.
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


@configclass
class AnymalDRoughPPOPfoRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D rough locomotion — vanilla MLP actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_rough_pfo"
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
class AnymalDRoughPPOPfoSimbaRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D rough locomotion — SimBa actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_d_rough_simba_pfo"
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
