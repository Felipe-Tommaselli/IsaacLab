# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimBa (Simple Baselines) architecture for RSL-RL.

Pre-LN residual block trunk that replaces the standard MLP body inside an
RSL-RL MLPModel.  Robot-agnostic: works for any observation/action dimension.
"""

import math

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


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
    """Drop-in RSL-RL MLP model config that routes to the shared SimBa model."""

    class_name: str = "isaaclab_tasks.utils.rsl_rl.models.simba.SimbaModel"
    simba_hidden_dim: int = 256
    simba_num_blocks: int = 2
    simba_final_gain: float = 1.0
    simba_depth_scale: bool = True
