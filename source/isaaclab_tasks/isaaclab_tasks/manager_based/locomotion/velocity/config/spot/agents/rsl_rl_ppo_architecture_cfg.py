# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modern-style RSL-RL runner configs for Spot flat (blind) locomotion.

The base Spot runner (rsl_rl_ppo_cfg.py / SpotFlatPPORunnerCfg) uses the legacy
RslRlPpoActorCriticCfg shape; these modern-style classes use the actor/critic +
obs_groups shape so SimBa/LinOp/PFO Hydra keys resolve.

max_iterations is set to 3000 here (vs Spot's 20000 baseline) for sweep
efficiency — three architectures × five seeds × 20000 iterations would burn
~16 hours per trial. Override per-sweep via the registry alias if a longer
budget is required.

Spot's tuned hyperparameters (entropy_coef=0.0025, value_loss_coef=0.5) are
preserved across all architecture variants.

The existing Isaac-Velocity-Flat-Spot-v0 task and SpotFlatPPORunnerCfg are
left untouched for backward compatibility.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils.rsl_rl.algorithms.ppo_pfo import RslRlPpoWithPfoCfg
from isaaclab_tasks.utils.rsl_rl.models.linop import RslRlLinOpModelCfg
from isaaclab_tasks.utils.rsl_rl.models.simba import RslRlSimbaModelCfg


@configclass
class SpotFlatPPOMLPRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Spot flat (blind) locomotion — modern-style MLP baseline."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 2000
    experiment_name = "spot_flat_mlp"
    store_code_state = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SpotFlatPPOSimbaRunnerCfg(SpotFlatPPOMLPRunnerCfg):
    """Spot flat locomotion — SimBa actor/critic."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "spot_flat_simba"
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
class SpotFlatPPOLinOpRunnerCfg(SpotFlatPPOMLPRunnerCfg):
    """Spot flat locomotion — MLP with LinOp factorized hidden layers."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "spot_flat_linop"
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
class SpotFlatPPOPfoRunnerCfg(SpotFlatPPOMLPRunnerCfg):
    """Spot flat locomotion — vanilla MLP actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "spot_flat_pfo"
        self.algorithm = RslRlPpoWithPfoCfg(
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0025,
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
class SpotFlatPPOPfoSimbaRunnerCfg(SpotFlatPPOMLPRunnerCfg):
    """Spot flat locomotion — SimBa actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "spot_flat_simba_pfo"
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
            value_loss_coef=0.5,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0025,
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
