# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import anymal

####################### SIMBA CONFIGS #######################


class _PreLNResidualBlock(nn.Module):
    """Pre-LN residual block with optional depth-stable branch scaling."""

    def __init__(self, hidden_dim: int, branch_scale: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.branch_scale = branch_scale
        nn.init.kaiming_normal_(self.linear1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.linear2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch_scale * self.linear2(torch.relu(self.linear1(self.norm(x))))


class _SimbaTrunk(nn.Module):
    """SimBa trunk: embedder -> N x PreLNResidualBlock -> post-LN -> output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        final_gain: float = 1.0,
        depth_scale: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}.")
        branch_scale = 1.0 / math.sqrt(num_blocks) if depth_scale and num_blocks > 0 else 1.0
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[_PreLNResidualBlock(hidden_dim, branch_scale=branch_scale) for _ in range(num_blocks)]
        )
        self.post_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        nn.init.orthogonal_(self.head.weight, gain=final_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.post_ln(self.blocks(self.embed(x))))


class SimbaModel(MLPModel):
    """RSL-RL MLPModel with the MLP body replaced by a SimBa trunk."""

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        simba_hidden_dim: int = 256,
        simba_num_blocks: int = 2,
        simba_final_gain: float = 1.0,
        simba_depth_scale: bool = True,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if simba_hidden_dim <= 0:
            raise ValueError(f"simba_hidden_dim must be positive, got {simba_hidden_dim}.")
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        if not isinstance(mlp_output_dim, int):
            raise ValueError(f"SimbaModel only supports scalar output dimensions, got {mlp_output_dim}.")
        self.mlp = _SimbaTrunk(
            self._get_latent_dim(),
            mlp_output_dim,
            simba_hidden_dim,
            simba_num_blocks,
            final_gain=simba_final_gain,
            depth_scale=simba_depth_scale,
        )


@configclass
class RslRlSimbaModelCfg(RslRlMLPModelCfg):
    """Drop-in RSL-RL MLP model config that routes to the local SimBa model."""

    class_name: str = (
        "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_simba_cfg.SimbaModel"
    )
    simba_hidden_dim: int = 256
    simba_num_blocks: int = 2
    simba_final_gain: float = 1.0
    simba_depth_scale: bool = True


#############################################################


@configclass
class AnymalDRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    experiment_name = "anymal_d_rough"
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
    )


@configclass
class AnymalDFlatPPORunnerCfg(AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]


@configclass
class AnymalDFlatPPOSimbaRunnerCfg(AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_simba"
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
class AnymalDFlatPPORunnerRecurrentCfg(AnymalDFlatPPORunnerCfg):
    actor = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class AnymalDFlatPPORunnerWithSymmetryCfg(AnymalDFlatPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    algorithm = RslRlPpoAlgorithmCfg(
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
        ),
    )


@configclass
class AnymalDRoughPPORunnerWithSymmetryCfg(AnymalDRoughPPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=anymal.compute_symmetric_states
        ),
    )
