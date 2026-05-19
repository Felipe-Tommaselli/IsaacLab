# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modern-style RSL-RL runner configs for H1 flat (blind) locomotion.

These configs use the actor/critic + obs_groups style (not the legacy
RslRlPpoActorCriticCfg style used in rsl_rl_ppo_cfg.py), which is required
for SimBa, LinOp, and PFO architecture variants that rely on class_name
resolution and MLPModel-compatible kwargs.

The existing Isaac-Velocity-Flat-H1-v0 task and H1FlatPPORunnerCfg are
left untouched for backward compatibility.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils.rsl_rl.algorithms.ppo_pfo import RslRlPpoWithPfoCfg
from isaaclab_tasks.utils.rsl_rl.models.linop import RslRlLinOpModelCfg
from isaaclab_tasks.utils.rsl_rl.models.simba import RslRlSimbaModelCfg


@configclass
class H1FlatPPOMLPRunnerCfg(RslRlOnPolicyRunnerCfg):
    """H1 flat (blind) locomotion — modern-style MLP baseline.

    Hyperparameters mirror H1's existing flat baseline (max_iterations=1000,
    entropy_coef=0.01) while using the actor/critic + obs_groups cfg shape
    so all architecture variants have compatible Hydra key paths.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "h1_flat_mlp"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
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
class H1FlatPPOSimbaRunnerCfg(H1FlatPPOMLPRunnerCfg):
    """H1 flat (blind) locomotion — SimBa actor/critic."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "h1_flat_simba"
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
class H1FlatPPOLinOpRunnerCfg(H1FlatPPOMLPRunnerCfg):
    """H1 flat (blind) locomotion — MLP with LinOp factorized hidden layers."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "h1_flat_linop"
        # Factored linear stack is sensitive to Adam's first-step stride.
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
class H1FlatPPOPfoRunnerCfg(H1FlatPPOMLPRunnerCfg):
    """H1 flat (blind) locomotion — vanilla MLP actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "h1_flat_pfo"
        self.algorithm = RslRlPpoWithPfoCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
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
class H1FlatPPOPfoSimbaRunnerCfg(H1FlatPPOMLPRunnerCfg):
    """H1 flat (blind) locomotion — SimBa actor/critic + PFO regularisation."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "h1_flat_simba_pfo"
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
            entropy_coef=0.01,
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
