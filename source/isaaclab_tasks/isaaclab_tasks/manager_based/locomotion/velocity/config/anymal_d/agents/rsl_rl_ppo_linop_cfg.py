# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Linear Over-Parameterization (LinOP) architecture variant for RSL-RL PPO.

Replaces each hidden square linear layer W with a product of k square factors
W_k @ ... @ W_1 (no nonlinearity between factors).  The function class is
identical to a single matrix — only the optimization landscape changes,
biasing gradient descent toward low-rank solutions (Huh et al., 2024).

Follows the same injection pattern as rsl_rl_ppo_simba_cfg.py.
"""

import math

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_simba_cfg import AnymalDRoughPPORunnerCfg


####################### LINOP COMPONENTS #######################


class _FactoredLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores W as a product of k square factors.

    Forward pass collapses the factors into a single matrix before applying,
    so runtime cost is one matmul (plus the factor-composition overhead which
    is negligible for typical hidden dims).

    Only supports square layers (in_features == out_features).
    """

    def __init__(self, dim: int, num_factors: int = 2, bias: bool = True) -> None:
        super().__init__()
        if num_factors < 2:
            raise ValueError(f"num_factors must be >= 2, got {num_factors}.")

        self.dim = dim
        self.num_factors = num_factors

        # Per-factor variance σ² = 2^(1/k) / d so the product's output
        # variance matches a standard Kaiming(ReLU) layer for any k.
        # This is the "constant output variance regardless of k" property
        # Huh et al. (Appendix F) flag as critical for stable training.
        std = math.sqrt(2.0 ** (1.0 / num_factors) / dim)
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, dim)) for _ in range(num_factors)]
        )
        for w in self.factors:
            nn.init.normal_(w, mean=0.0, std=std)

        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

        # Eval-time cache for the collapsed weight.
        self.register_buffer("_cached_weight", torch.empty(0), persistent=False)
        self._cache_valid = False

    def _collapsed_weight(self) -> torch.Tensor:
        """Compute the effective weight matrix W_e = W_k @ ... @ W_1."""
        W = self.factors[0]
        for i in range(1, self.num_factors):
            W = self.factors[i] @ W
        return W

    def train(self, mode: bool = True):
        self._cache_valid = False
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            W = self._collapsed_weight()
        else:
            if not self._cache_valid:
                with torch.no_grad():
                    self._cached_weight = self._collapsed_weight().detach()
                self._cache_valid = True
            W = self._cached_weight
        return nn.functional.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_factors={self.num_factors}, bias={self.bias is not None}"


class LinOpModel(MLPModel):
    """RSL-RL MLPModel with hidden square linear layers replaced by factored products.

    After the parent MLPModel builds the standard MLP body, this class walks
    the network and swaps every square nn.Linear in the hidden layers with a
    _FactoredLinear of the same dimension.  Non-square layers (input projection,
    output head) are left untouched.
    """

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
        linop_num_factors: int = 2,
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

        if linop_num_factors < 2:
            raise ValueError(f"linop_num_factors must be >= 2, got {linop_num_factors}.")

        # Walk self.mlp and replace square nn.Linear modules with _FactoredLinear.
        # self.mlp is an nn.Sequential of (Linear, Activation, Linear, Activation, ..., Linear).
        # We replace all hidden Linear layers (all except the last one in the sequential).
        linear_indices = [i for i, m in enumerate(self.mlp) if isinstance(m, nn.Linear)]
        # All except the last linear are "hidden" layers
        hidden_linear_indices = linear_indices[:-1] if len(linear_indices) > 1 else []

        for idx in hidden_linear_indices:
            layer = self.mlp[idx]
            if layer.in_features == layer.out_features:
                self.mlp[idx] = _FactoredLinear(
                    dim=layer.in_features,
                    num_factors=linop_num_factors,
                    bias=layer.bias is not None,
                )


@configclass
class RslRlLinOpModelCfg(RslRlMLPModelCfg):
    """Drop-in RSL-RL MLP model config that routes to the local LinOp model."""

    class_name: str = (
        "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_linop_cfg.LinOpModel"
    )
    linop_num_factors: int = 2


#############################################################


@configclass
class AnymalDFlatPPOLinOpRunnerCfg(AnymalDRoughPPORunnerCfg):
    """Anymal-D flat locomotion — MLP-3 with LinOP factorized hidden layers."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat_linop"
        self.actor = RslRlLinOpModelCfg(
            hidden_dims=[512, 512, 512, 512, 256],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
            linop_num_factors=2,
        )
        self.critic = RslRlLinOpModelCfg(
            hidden_dims=[512, 512, 512, 512, 256],
            activation="elu",
            obs_normalization=False,
            linop_num_factors=2,
        )
