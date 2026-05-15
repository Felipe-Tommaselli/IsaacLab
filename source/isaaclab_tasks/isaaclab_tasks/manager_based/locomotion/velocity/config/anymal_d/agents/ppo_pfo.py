# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proximal Feature Optimization (PFO) – Moala et al.

Drop-in PPO subclass that adds a feature-drift regularization loss on the
actor's pre-activations.  Zero modifications to rsl_rl are required.

Architecture notes
------------------
SimBa (_SimbaTrunk): embed → blocks → post_ln → head
  - PFO target: INPUT to post_ln  (= raw residual-block output, before LayerNorm)
  - Hooking post-LN output (= head input) would make PFO near-no-op because
    LayerNorm already constrains magnitude.

Vanilla MLP (nn.Sequential):
  - PFO target: OUTPUT of last hidden Linear  (= true pre-activation before ReLU/tanh)

Batch-indexing strategy
-----------------------
We deepcopy only the _SimbaTrunk (≈2 MB for hidden_dim=128) ONCE at the
start of update(), before any gradient step.  Inside each mini-batch we:
  1. Register two hooks on the *current* trunk during its normal PPO forward:
       - pre-hook on trunk itself  → captures latent (post-normaliser input)
       - pre-hook on trunk.post_ln → captures current pre-activation (with grad)
  2. Run old_trunk(latent.detach()) with no_grad → old pre-activation

This avoids all index-tracking across the shuffled mini-batch generator.
"""

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO

from .rsl_rl_ppo_linop_cfg import _FactoredLinear


# ---------------------------------------------------------------------------
# Architecture-agnostic hook adapter
# ---------------------------------------------------------------------------


def _get_hook_targets(trunk: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Return (trunk_module, preact_module) for the given trunk.

    A forward-pre-hook on *trunk_module* captures the latent input fed to the
    trunk (post-normaliser, pre-MLP).  The preact_module is used to capture
    the true penultimate pre-activation.

    Returns
    -------
    trunk_module
        The top-level trunk (same object passed in).  Pre-hook captures the
        latent fed to the trunk.
    preact_module
        The submodule used to capture the penultimate pre-activation:
        - SimBa: ``trunk.post_ln`` — a *pre-hook* on this captures
          ``blocks(embed(x))``, the true pre-activation before LayerNorm.
        - Vanilla MLP (nn.Sequential): the last hidden ``nn.Linear`` (the one
          *before* the output head) — a *forward hook* on this captures its
          output, which is the true pre-activation before the subsequent
          activation function.  Note: for MLP the capture must use a forward
          hook (output), not a pre-hook (input).
    """
    # SimBa (_SimbaTrunk) has a post_ln attribute
    if hasattr(trunk, "post_ln"):
        return trunk, trunk.post_ln

    # Vanilla MLP exposed as nn.Sequential — find the last hidden nn.Linear
    # (second-to-last Linear overall; the last Linear is the output head).
    if isinstance(trunk, nn.Sequential):
        linears = [(i, m) for i, m in enumerate(trunk)
                   if isinstance(m, (nn.Linear, _FactoredLinear))]
        if len(linears) >= 2:
            # Second-to-last Linear = last hidden Linear
            return trunk, linears[-2][1]
        # Fallback: only one Linear (degenerate MLP), hook its input
        return trunk, trunk[-1]

    raise ValueError(
        f"Unsupported trunk type {type(trunk).__name__}. "
        "Add an entry in _get_hook_targets() for your architecture."
    )


def _register_capture_hooks(trunk: nn.Module, preact_mod: nn.Module, *, use_forward_hook: bool = False) -> tuple:
    """Register temporary hooks; return (handles, captured_dict).

    captured["latent"]  : input to trunk (post-normaliser)
    captured["preact"]  : pre-activation from preact_mod

    For SimBa, preact_mod is trunk.post_ln and we use a pre-hook (input =
    true pre-act).  For vanilla MLP, preact_mod is the last hidden Linear
    and we use a forward hook (output = true pre-act before activation fn).
    Set ``use_forward_hook=True`` for the MLP case.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook_latent(module, inputs):
        captured["latent"] = inputs[0]

    def _hook_preact_pre(module, inputs):
        captured["preact"] = inputs[0]

    def _hook_preact_fwd(module, inputs, output):
        captured["preact"] = output

    h1 = trunk.register_forward_pre_hook(_hook_latent)
    if use_forward_hook:
        h2 = preact_mod.register_forward_hook(_hook_preact_fwd)
    else:
        h2 = preact_mod.register_forward_pre_hook(_hook_preact_pre)
    return (h1, h2), captured


def _register_preact_hook_only(preact_mod: nn.Module, *, use_forward_hook: bool = False) -> tuple:
    """Register a single hook on preact_mod; return (handle, captured).

    Uses a pre-hook (captures input) by default, or a forward hook (captures
    output) when ``use_forward_hook=True`` (needed for MLP last-hidden-Linear).
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook_pre(module, inputs):
        captured["preact"] = inputs[0]

    def _hook_fwd(module, inputs, output):
        captured["preact"] = output

    if use_forward_hook:
        handle = preact_mod.register_forward_hook(_hook_fwd)
    else:
        handle = preact_mod.register_forward_pre_hook(_hook_pre)
    return handle, captured


# ---------------------------------------------------------------------------
# PPO + PFO
# ---------------------------------------------------------------------------


class PPOWithPFO(PPO):
    """Proximal Policy Optimization with Proximal Feature Optimization.

    Adds ``pfo_coef * E[(h_θ(x) - h_θ_old(x))²]`` to the PPO loss, where
    ``h`` is the actor's penultimate pre-activation (input to post_ln for
    SimBa; input to output head for vanilla MLPs).

    The old pre-activations are obtained by running a lightweight copy of the
    actor trunk (frozen at the start of each ``update()`` call, before any
    gradient step) on the same mini-batch observations.  This matches the
    exact definition in Moala et al. where θ_old is the rollout-collection
    policy.

    Parameters
    ----------
    pfo_coef : float
        Weight of the PFO regularisation term.  Moala et al. recommend the
        nearest power-of-10 matching the PPO surrogate-loss magnitude.
        For SimBa locomotion tasks start with 1.0 and sweep {0.1, 1.0, 10.0}.
    pfo_all_layers : bool
        If True, regularise the outputs of *all* residual blocks (SimBa) or
        all hidden Linear outputs (MLP), not only the penultimate layer.
        Corresponds to the "Regularize all pre-activations" ablation in the
        paper.  Defaults to False.
    """

    def __init__(self, *args, pfo_coef: float = 1.0, pfo_all_layers: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pfo_coef = pfo_coef
        self.pfo_all_layers = pfo_all_layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_actor_trunk(self) -> nn.Module:
        """Return the MLP/trunk inside the actor model."""
        # SimbaModel stores it as actor.mlp; MLPModel does the same.
        # Use self.actor (the active module in the forward path) rather than
        # self._raw_actor, so hooks attach correctly even under torch.compile.
        return self.actor.mlp  # type: ignore[attr-defined]

    def _pfo_loss_single(
        self,
        trunk: nn.Module,
        old_trunk: nn.Module,
        preact_mod: nn.Module,
        old_preact_mod: nn.Module,
        latent_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PFO loss for the penultimate layer.

        latent_batch is captured during the PPO actor forward (with grad).
        old_trunk is run with no_grad on the same latent.
        """
        # Old pre-activation (no grad needed)
        old_handle, old_cap = _register_preact_hook_only(old_preact_mod)
        with torch.no_grad():
            old_trunk(latent_batch.detach())
        old_handle.remove()

        # Current pre-activation was already captured during PPO actor forward.
        # Re-run trunk to capture it with gradient, using the latent already
        # produced by the actor's normaliser (captured via hook).
        curr_handle, curr_cap = _register_preact_hook_only(preact_mod)
        trunk(latent_batch)  # lightweight re-run through trunk only
        curr_handle.remove()

        return self.pfo_coef * (curr_cap["preact"] - old_cap["preact"].detach()).pow(2).mean()

    def _pfo_loss_all_layers_simba(
        self,
        trunk: nn.Module,
        old_trunk: nn.Module,
        latent_batch: torch.Tensor,
    ) -> torch.Tensor:
        """PFO over all SimBa residual-block outputs + post_ln input."""
        curr_feats: list[torch.Tensor] = []
        old_feats: list[torch.Tensor] = []

        def _make_hook(store: list):
            def _h(mod, inputs, output):
                store.append(output)
            return _h

        curr_handles = [b.register_forward_hook(_make_hook(curr_feats)) for b in trunk.blocks]  # type: ignore
        old_handles = [b.register_forward_hook(_make_hook(old_feats)) for b in old_trunk.blocks]  # type: ignore

        # Also include post_ln input (penultimate pre-act)
        curr_preact_h, curr_preact_cap = _register_preact_hook_only(trunk.post_ln)       # type: ignore
        old_preact_h, old_preact_cap = _register_preact_hook_only(old_trunk.post_ln)     # type: ignore

        with torch.no_grad():
            old_trunk(latent_batch.detach())
        trunk(latent_batch)

        for h in curr_handles + old_handles:
            h.remove()
        curr_preact_h.remove()
        old_preact_h.remove()

        # Add penultimate pre-act to feature lists
        curr_feats.append(curr_preact_cap["preact"])
        old_feats.append(old_preact_cap["preact"].detach())

        num_sites = len(curr_feats)
        loss = sum(
            (c - o.detach()).pow(2).mean()
            for c, o in zip(curr_feats, old_feats)
        )
        # Normalize by number of hook sites so pfo_coef has the same
        # effective scale regardless of single-layer vs all-layers mode.
        return self.pfo_coef * loss / max(num_sites, 1)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:  # noqa: C901
        """Full PPO update loop with PFO regularisation injected per mini-batch.

        Faithfully replicates PPO.update() (adaptive LR, clipped value loss,
        grad clipping) and adds ``pfo_loss`` to the total loss scalar before
        backward().  Symmetry and RND extensions are preserved.
        """
        if self.pfo_coef <= 0.0:
            return super().update()

        trunk = self._get_actor_trunk()
        _, preact_mod = _get_hook_targets(trunk)
        # MLP needs forward hooks (output capture), SimBa uses pre-hooks
        _fwd_hook = not hasattr(trunk, "post_ln")

        # Snapshot theta_old trunk ONCE before any gradient step (~2 MB for
        # hidden_dim=128).  Only the trunk weights are copied; normaliser state
        # lives in self.actor and is NOT copied (it's shared, read-only here).
        old_trunk = copy.deepcopy(trunk).eval()
        _, old_preact_mod = _get_hook_targets(old_trunk)

        # ---- standard PPO accumulators ----
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_pfo_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # ----------------------------------------------------------
            # Actor forward – we hook the trunk to capture the latent
            # (post-normaliser input to trunk) AND the current preact.
            # Both are captured during this single actor forward so we
            # pay zero extra cost for PFO capture.
            # ----------------------------------------------------------
            (h_latent, h_preact), cap = _register_capture_hooks(
                trunk, preact_mod, use_forward_hook=_fwd_hook
            )
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            h_latent.remove()
            h_preact.remove()

            latent_batch: torch.Tensor = cap["latent"]        # (B, D_latent) with grad
            curr_preact: torch.Tensor = cap["preact"]         # (B, D_hidden) with grad

            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            # ----------------------------------------------------------
            # Adaptive KL learning-rate schedule (verbatim from PPO)
            # ----------------------------------------------------------
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ----------------------------------------------------------
            # Surrogate loss
            # ----------------------------------------------------------
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ----------------------------------------------------------
            # Value loss
            # ----------------------------------------------------------
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # ----------------------------------------------------------
            # PFO loss
            # Old preact: run old_trunk on the SAME latent (detached) that
            # the actor's normaliser already produced.  This is correct because
            # the normaliser state at update() start == theta_old's normaliser.
            # ----------------------------------------------------------
            if self.pfo_all_layers and hasattr(trunk, "blocks"):
                pfo_loss = self._pfo_loss_all_layers_simba(trunk, old_trunk, latent_batch)
            else:
                # Old preact
                old_h, old_cap = _register_preact_hook_only(
                    old_preact_mod, use_forward_hook=_fwd_hook
                )
                with torch.no_grad():
                    old_trunk(latent_batch.detach())
                old_h.remove()
                old_preact = old_cap["preact"].detach()
                pfo_loss = self.pfo_coef * (curr_preact - old_preact).pow(2).mean()

            # ----------------------------------------------------------
            # Total loss
            # ----------------------------------------------------------
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + pfo_loss
            )

            # RND loss
            rnd_loss = (
                self.rnd.compute_loss(batch.observations[:original_batch_size])
                if self.rnd
                else None
            )

            # Symmetry loss
            symmetry_loss = None
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # ----------------------------------------------------------
            # Backward + grad clipping + optimiser step
            # ----------------------------------------------------------
            self.optimizer.zero_grad()
            loss.backward()
            if rnd_loss is not None:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if rnd_loss is not None:
                self.rnd.optimizer.step()

            # ----------------------------------------------------------
            # Accumulate statistics
            # ----------------------------------------------------------
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_pfo_loss += pfo_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ---- normalise ----
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "pfo": mean_pfo_loss / num_updates,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates

        self.storage.clear()
        return loss_dict
