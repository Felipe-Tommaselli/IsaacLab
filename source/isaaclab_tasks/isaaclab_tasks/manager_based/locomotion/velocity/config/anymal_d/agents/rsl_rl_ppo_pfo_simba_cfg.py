# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner configs for Anymal-D + SimBa + Proximal Feature Optimization (PFO).

Imports SimbaModel / RslRlSimbaModelCfg from the existing SimBa config to
avoid any code duplication.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import AnymalDFlatPPORunnerCfg
from .rsl_rl_ppo_simba_cfg import AnymalDRoughPPORunnerCfg, RslRlSimbaModelCfg


# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------


@configclass
class RslRlPpoWithPfoCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with PFO regularisation parameters.

    ``class_name`` points to the local PPOWithPFO subclass so that
    rsl_rl's ``resolve_callable`` / ``construct_algorithm`` picks it up
    without any modifications to the library.
    """

    class_name: str = (
        "isaaclab_tasks.manager_based.locomotion.velocity"
        ".config.anymal_d.agents.ppo_pfo.PPOWithPFO"
    )

    pfo_coef: float = 1.0
    """PFO regularisation coefficient.

    Moala et al. recommend the nearest power-of-10 that matches the
    magnitude of the PPO surrogate loss.  For SimBa locomotion tasks
    a good starting sweep is {0.1, 1.0, 10.0}.
    """

    pfo_all_layers: bool = False
    """If True, regularise all residual-block outputs (SimBa) / hidden
    layer outputs (MLP), not only the penultimate pre-activation.
    Corresponds to the 'Regularize all pre-activations' ablation in the
    paper.
    """


# ---------------------------------------------------------------------------
# Runner configs
# ---------------------------------------------------------------------------


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

        # Actor: SimBa with empirical_normalization (obs_normalization=True)
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

        # Critic: SimBa with wider hidden dim (no distribution head)
        self.critic = RslRlSimbaModelCfg(
            hidden_dims=[128, 128, 128],
            activation="relu",
            obs_normalization=True,
            simba_hidden_dim=256,
            simba_num_blocks=2,
            simba_final_gain=1.0,
            simba_depth_scale=True,
        )

        # Algorithm: PPO + PFO
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
