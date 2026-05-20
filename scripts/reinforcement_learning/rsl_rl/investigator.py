"""
investigator.py — Self-contained observability engine for legged locomotion rank analysis.

Integration (in train.py):

    from investigator import Investigator, InvestigatorCfg

    investigator = Investigator(runner, cfg=InvestigatorCfg())
    investigator.install(sort_fn=lambda obs: obs[:, 9:12].norm(dim=-1))
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    investigator.generate_report()

Designed for rsl_rl with IsaacLab. Actor/Critic accessed via runner.alg.policy.actor/.critic.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from functools import wraps
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

_CMAP_BWY = LinearSegmentedColormap.from_list(
    'blue_white_yellow', ['#cc0000', 'white', '#FFB300'])
_CMAP_WY = LinearSegmentedColormap.from_list(
    'white_yellow', ['white', '#FFB300'])
_CMAP_WB = LinearSegmentedColormap.from_list(
    'white_blue', ['white', '#1565C0'])


def _gram_colornorm(g_np):
    """3-color norm/cmap for diagnostic Gram view."""
    vmin, vmax = g_np.min(), g_np.max()
    if vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), _CMAP_BWY
    return None, _CMAP_WY


def _gram_pattern(g_np, cmap=None):
    """Clip negatives -> white, sequential colormap. Pattern view (Hu et al.)."""
    g = np.clip(g_np, 0, None)
    vmax = g.max() if g.max() > 0 else 1.0
    return g, None, (cmap or _CMAP_WY), 0.0, vmax


_SUP_DIGITS = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _setup_sci_xaxis(ax, values):
    """Configure x-axis with scientific notation: ticks as coefficients, exponent in label."""
    from matplotlib.ticker import FuncFormatter
    max_val = max(values) if values else 0
    if max_val <= 0:
        ax.set_xlabel("Total env steps")
        return
    exp = int(np.floor(np.log10(max_val)))
    divisor = 10 ** exp
    exp_str = str(exp).translate(_SUP_DIGITS)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: f"{x / divisor:.1f}"))
    ax.set_xlabel(f"Total env steps (\u00d710{exp_str})")


def _fmt_total_steps(total_steps: int | float) -> str:
    """Format total env steps for subplot titles (e.g. '1.97\u00d710\u2078')."""
    total_steps = int(total_steps)
    if total_steps == 0:
        return "step 0"
    exp = int(np.floor(np.log10(abs(total_steps))))
    coeff = total_steps / 10 ** exp
    exp_str = str(exp).translate(_SUP_DIGITS)
    return f"{coeff:.2f}\u00d710{exp_str} steps"


# [WANDB INTEGRATION] soft-import; all wandb calls below are gated on _WANDB_AVAILABLE
_WANDB_AVAILABLE = False
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    pass

# [MLFLOW INTEGRATION] soft-import; mirror of the wandb pattern
_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Semantic labels (Spot-specific)
# ===================================================================

# Action indices: FL(hx,hy,kn), FR(hx,hy,kn), HL(hx,hy,kn), HR(hx,hy,kn)
ACTION_LABELS = [
    "FL_hx", "FL_hy", "FL_kn",
    "FR_hx", "FR_hy", "FR_kn",
    "HL_hx", "HL_hy", "HL_kn",
    "HR_hx", "HR_hy", "HR_kn",
]

# Observation groups: (name, start_idx, end_idx_exclusive)
OBS_GROUPS = [
    ("base_vel", 0, 3),
    ("ang_vel", 3, 6),
    ("gravity", 6, 9),
    ("cmd", 9, 13),
    ("joint_pos", 13, 25),
    ("joint_vel", 25, 37),
    ("last_act", 37, 49),
]


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class InvestigatorCfg:
    """All knobs for the Investigator."""

    enabled: bool = True

    # -- Backend --
    backend: str = "wandb"
    """Experiment tracker: "wandb" or "mlflow". The wandb_* toggles below
    are semantic 'log this thing?' switches that apply to both backends."""

    # -- Cadence --
    log_interval: int = 100
    """Compute live representation metrics every N training iterations."""
    checkpoint_interval: int = 500
    """Compute expensive metrics (Gram, Jacobian, native plots) every N iterations."""
    weight_rank_interval: int = 250
    """Compute per-layer weight erank every N iterations."""

    # -- Eval observations --
    n_eval_obs: int = 2048
    """Fixed eval observation count for Gram / Jacobian analysis."""

    # -- Jacobian --
    jacobian_batch_size: int = 64
    """Subset of eval obs for Jacobian erank (O(action_dim) backward passes per sample)."""
    jacobian_use_vmap: bool = True
    """Use torch.func.jacrev + vmap (PyTorch >= 2.0)."""

    # -- Saving (local disk backup) --
    save_features: bool = True
    save_grams: bool = True
    save_spectra: bool = True
    save_jacobian_spectra: bool = True
    save_jacobians: bool = True
    """Save full J_mean matrix (not just SVs) for post-hoc heatmaps."""

    # -- WandB --
    wandb_log: bool = True
    wandb_gram_image: bool = True
    wandb_feature_heatmap: bool = True
    """Log feature activation heatmap (N x D) as wandb.Image."""
    wandb_jacobian_heatmap: bool = True
    """Log Jacobian heatmap (action_dim x obs_dim) as wandb.Image."""
    wandb_weight_heatmap: bool = True
    """Log per-layer weight heatmaps as wandb.Image."""
    wandb_native_plots: bool = False
    """Deprecated — wandb.plot.* renders as raw tables in recent wandb versions."""
    wandb_spectral_histograms: bool = True
    """Log wandb.Histogram for SV distributions (produces heatmap-over-time)."""
    wandb_artifact_on_end: bool = True
    """Upload investigator/ dir as wandb Artifact at end of training."""
    wandb_summary_on_end: bool = True
    """Write final summary metrics to wandb.run.summary."""

    # -- Plotting (matplotlib fallback) --
    plot_format: str = "png"
    plot_dpi: int = 150
    figsize_gram: tuple[float, float] = (4.0, 4.0)
    figsize_evolution: tuple[float, float] = (14.0, 5.0)

    # -- Thresholds --
    dead_neuron_threshold: float = 1e-1
    erank_eps: float = 1e-10

    # -- Environment step aggregation (fallback; auto-detected at install) --
    num_steps_per_env: int = 24
    """Rollout horizon per env per iteration (fallback if auto-detect fails)."""
    num_envs: int = 4096
    """Number of parallel environments (fallback if auto-detect fails)."""

    # -- Scrape from rsl_rl's learn() locals for post-hoc comparison --
    # Scalars (float/int) go as-is. Deques/lists are reduced with mean.
    # Dicts are flattened one level (e.g. loss_dict -> investigator/train/value_loss).
    locs_scalar_keys: tuple[str, ...] = (
        "rewbuffer",         # deque of episode rewards -> mean_reward
        "lenbuffer",         # deque of episode lengths -> mean_episode_length
        "loss_dict",         # dict: surrogate_loss, value_loss, entropy, ...
        "collection_time",
        "learn_time",
        "ep_infos",          # list of dicts: custom episode metrics (rsl_rl < 5.0)
        "extra_ep_info_buffer", # list of dicts: custom episode metrics (rsl_rl >= 5.0)
    )


# ===================================================================
# Math utilities
# ===================================================================

def effective_rank(S: torch.Tensor, eps: float = 1e-10) -> float:
    """erank(A) = exp(-sum s_bar_i log s_bar_i) where s_bar_i = sigma_i / sum sigma_j."""
    S = S[S > eps]
    if len(S) == 0:
        return 0.0
    S_norm = S / S.sum()
    entropy = -(S_norm * torch.log(S_norm + eps)).sum()
    return torch.exp(entropy).item()


def pca_rank(S: torch.Tensor, threshold: float = 0.99) -> int:
    """Smallest k: top-k singular values explain >= threshold variance."""
    var = S ** 2
    explained = torch.cumsum(var, dim=0) / var.sum()
    return torch.searchsorted(explained, threshold).item() + 1


def compute_gram_cosine(features: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity Gram matrix [N, D] -> [N, N]."""
    F_norm = F.normalize(features, dim=-1)
    return F_norm @ F_norm.T


def _to_scalar(v) -> float | None:
    """Best-effort cast to a Python float.

    - numbers -> float
    - deque/list/tuple of numbers -> mean (None if empty)
    - 0-d / 1-d torch tensors -> mean item
    - anything else -> None
    """
    if v is None:
        return None
    scalar = _normalize_scalar(v)
    if scalar is not None:
        return scalar
    if isinstance(v, torch.Tensor):
        if v.numel() == 0:
            return None
        return _normalize_scalar(v.float().mean())
    # deque, list, tuple
    try:
        seq = list(v)
    except TypeError:
        return None
    if not seq:
        return None
    try:
        return _normalize_scalar(np.mean([float(x) for x in seq]))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _MediaMetric:
    """Marker for backend media objects that should not enter scalar history."""

    value: object


def _normalize_scalar(value) -> float | None:
    """Return a finite Python float for scalar metric values, otherwise None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        scalar = float(value)
    elif isinstance(value, np.generic):
        scalar = float(value.item())
    elif isinstance(value, torch.Tensor):
        if value.ndim != 0:
            return None
        scalar = float(value.detach().cpu().item())
    else:
        return None
    return scalar if math.isfinite(scalar) else None


def _split_metric_payload(metrics: dict) -> tuple[dict[str, float], dict]:
    """Separate finite scalars from explicitly marked media objects."""
    scalars = {}
    media = {}
    for key, value in metrics.items():
        scalar = _normalize_scalar(value)
        if scalar is not None:
            scalars[key] = scalar
        elif isinstance(value, _MediaMetric):
            media[key] = value.value
    return scalars, media


def _coerce_scalar_payload(payload: dict) -> dict[str, float]:
    return {k: scalar for k, v in payload.items()
            if (scalar := _normalize_scalar(v)) is not None}


# ===================================================================
# Backend adapters (wandb / mlflow)
# ===================================================================

def _make_histogram_fig(arr, name: str, bins: int = 64):
    """MLflow has no native Histogram type; render via matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.asarray(arr), bins=bins)
    ax.set_title(name)
    fig.tight_layout()
    return fig


class _WandbBackend:
    """W&B adapter with scalar and media payloads logged separately."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._metrics_defined = False

    def is_active(self) -> bool:
        return _WANDB_AVAILABLE and wandb.run is not None

    def _define_metrics(self):
        if self._metrics_defined or not self.is_active():
            return
        try:
            wandb.define_metric("total_env_steps")
            wandb.define_metric("investigator/*",
                                step_metric="total_env_steps")
            self._metrics_defined = True
        except Exception as e:
            warnings.warn(f"[Investigator] wandb metric definition failed: {e}")

    def log_params(self, d: dict):
        self._define_metrics()
        wandb.config.update({"investigator": d}, allow_val_change=True)

    def add_histogram(self, metrics: dict, name: str, arr, iteration: int):
        metrics[name] = _MediaMetric(wandb.Histogram(np.asarray(arr)))

    def add_image(self, metrics: dict, name: str, fig, iteration: int,
                  subdir: str = "plots"):
        metrics[name] = _MediaMetric(wandb.Image(fig))

    def flush(self, scalars: dict, media: dict, iteration: int):
        self._define_metrics()
        scalar_payload = _coerce_scalar_payload(scalars)
        if scalar_payload:
            wandb.log({**scalar_payload, "total_env_steps": iteration})
        if media:
            wandb.log({**media, "total_env_steps": iteration})

    def log_summary(self, summary: dict):
        wandb.run.summary.update(_coerce_scalar_payload(summary))

    def log_artifacts(self, log_dir, name: str):
        art = wandb.Artifact(name, type="analysis")
        art.add_dir(str(log_dir))
        wandb.log_artifact(art)


class _MlflowBackend:
    """Same surface mapped to MLflow: figures inline via log_figure, scalars batched in flush."""

    def __init__(self, cfg):
        self.cfg = cfg

    def is_active(self) -> bool:
        return _MLFLOW_AVAILABLE and mlflow.active_run() is not None

    def log_params(self, d: dict):
        mlflow.log_params({f"investigator.{k}": str(v) for k, v in d.items()})

    def add_histogram(self, metrics: dict, name: str, arr, iteration: int):
        fig = _make_histogram_fig(arr, name)
        safe = name.replace("/", "_")
        mlflow.log_figure(fig, f"histograms/{safe}_step_{iteration}.png")
        plt.close(fig)

    def add_image(self, metrics: dict, name: str, fig, iteration: int,
                  subdir: str = "plots"):
        safe = name.replace("/", "_")
        mlflow.log_figure(fig, f"{subdir}/{safe}_step_{iteration}.png")

    def flush(self, scalars: dict, media: dict, iteration: int):
        scalars = _coerce_scalar_payload(scalars)
        if scalars:
            mlflow.log_metrics({**scalars, "total_env_steps": float(iteration)},
                               step=iteration)

    def log_summary(self, summary: dict):
        flat = _coerce_scalar_payload(summary)
        if flat:
            mlflow.log_metrics(flat)

    def log_artifacts(self, log_dir, name: str):
        mlflow.log_artifacts(str(log_dir), artifact_path="investigator")


class _MultiBackend:
    """Fan-out to multiple sub-backends; each gates itself via ``is_active``.

    Lets wandb and mlflow receive the same Investigator stream in parallel.
    Sub-backends that report ``is_active() == False`` are silently skipped, so
    the same configuration works whether one or both trackers are running.
    """

    def __init__(self, cfg):
        self._backends = [_WandbBackend(cfg), _MlflowBackend(cfg)]

    def _active(self):
        return [b for b in self._backends if b.is_active()]

    def is_active(self) -> bool:
        return any(b.is_active() for b in self._backends)

    def log_params(self, d: dict):
        for b in self._active():
            b.log_params(d)

    def add_histogram(self, metrics: dict, name: str, arr, iteration: int):
        for b in self._active():
            b.add_histogram(metrics, name, arr, iteration)

    def add_image(self, metrics: dict, name: str, fig, iteration: int,
                  subdir: str = "plots"):
        for b in self._active():
            b.add_image(metrics, name, fig, iteration, subdir=subdir)

    def flush(self, scalars: dict, media: dict, iteration: int):
        for b in self._active():
            b.flush(scalars, media, iteration)

    def log_summary(self, summary: dict):
        for b in self._active():
            b.log_summary(summary)

    def log_artifacts(self, log_dir, name: str):
        for b in self._active():
            b.log_artifacts(log_dir, name)


# ===================================================================
# Layer discovery
# ===================================================================

def _is_linear_like(m: nn.Module) -> bool:
    """True for nn.Linear and any drop-in replacement exposing a 2D .weight.

    Catches _FactoredLinear (and future variants) without importing them.
    Excludes nn.ParameterList, LayerNorm, BatchNorm, etc.
    """
    if isinstance(m, nn.Linear):
        return True
    w = getattr(m, "weight", None)
    return (w is not None
            and isinstance(w, torch.Tensor)
            and w.ndim == 2
            and not isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)))


def _find_linear_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if _is_linear_like(m)]


def _find_penultimate_linear(model: nn.Module) -> tuple[str, nn.Module]:
    linears = _find_linear_layers(model)
    if len(linears) < 2:
        raise ValueError(f"Need >= 2 Linear layers, found {len(linears)}.")
    return linears[-2]


# ===================================================================
# The Investigator
# ===================================================================

class Investigator:
    """Self-contained observability engine for representation rank analysis.

    Hooks into rsl_rl's OnPolicyRunner via monkey-patching.
    All metrics go to wandb; local disk is a backup.
    """

    def __init__(self, runner, cfg: InvestigatorCfg | None = None):
        self.runner = runner
        self.cfg = cfg or InvestigatorCfg()
        self.device = runner.device

        self._fixed_eval_obs: torch.Tensor | None = None
        self._sort_indices: torch.Tensor | None = None
        self._phase_fn: callable | None = None
        self._log_dir: Path | None = None
        self._history: dict[str, list] = {}
        self._installed = False
        self._backend = None
        self._step_scale: int = 1
        self._num_learning_iterations: int | None = None
        self._half_dumped = False
        self._scalar_writer = None
        self._prev_eval_actions: torch.Tensor | None = None

    # ---------------------------------------------------------
    # Installation
    # ---------------------------------------------------------

    def install(self, sort_fn: callable | None = None,
                phase_fn: callable | None = None):
        """Monkey-patch runner.learn() to inject analysis hooks.

        Args:
            sort_fn: Optional callable(obs_batch) -> 1D sort keys.
                     Sorts Gram matrix rows for visible block structure.
                     Example: lambda obs: obs[:, 9:12].norm(dim=-1)
            phase_fn: Optional callable(env) -> Tensor[int] of shape [num_envs].
                      Returns a phase label per environment for phase-conditioned
                      Jacobian analysis. Labels re-queried each checkpoint.
                      Example: lambda env: env.scene["contact_sensor"].data
                               .net_forces_w[:, :, 2].gt(1.0).sum(dim=-1)
        """
        if not self.cfg.enabled or self._installed:
            return

        self._phase_fn = phase_fn

        runner = self.runner

        # Auto-detect environment step aggregation parameters
        try:
            _nsteps = runner.cfg.get("num_steps_per_env", self.cfg.num_steps_per_env)
            _nenvs = getattr(runner.env, "num_envs", self.cfg.num_envs)
            self._step_scale = int(_nsteps) * int(_nenvs)
        except Exception:
            self._step_scale = self.cfg.num_steps_per_env * self.cfg.num_envs
        print(f"[Investigator] Step scale: {self._step_scale} "
              f"(num_steps_per_env \u00d7 num_envs)")

        # Output directory
        log_dir = getattr(runner, "log_dir", None) or "logs/investigator_default"
        self._log_dir = Path(log_dir) / "investigator"
        for sub in ["features", "grams", "jacobians", "spectra", "plots", "ppo"]:
            (self._log_dir / sub).mkdir(parents=True, exist_ok=True)

        # Collect fixed eval observations
        self._collect_fixed_eval_obs()

        # Sort indices
        if sort_fn is not None:
            with torch.no_grad():
                keys = sort_fn(self._fixed_eval_obs.to(self.device))
                self._sort_indices = torch.argsort(keys).cpu()

        # Save config locally
        with open(self._log_dir / "investigator_cfg.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2, default=str)

        # Backend selection (wandb, mlflow, or both via "multi")
        if self.cfg.backend == "multi":
            self._backend = _MultiBackend(self.cfg)
        elif self.cfg.backend == "mlflow":
            self._backend = _MlflowBackend(self.cfg)
        else:
            self._backend = _WandbBackend(self.cfg)

        # Push investigator config to the active run
        if self.cfg.wandb_log and self._backend.is_active():
            self._backend.log_params(asdict(self.cfg))

        # -- Monkey-patch learn() --
        original_learn = runner.learn
        investigator = self

        @wraps(original_learn)
        def patched_learn(num_learning_iterations: int,
                          init_at_random_ep_len: bool = False):
            investigator._num_learning_iterations = num_learning_iterations
            investigator._half_dumped = False
            # rsl_rl >= 5.0: logging goes through runner.logger.log(); no runner.log()
            logger = getattr(runner, "logger", None)
            if logger is not None and not hasattr(runner, "log"):
                original_log = logger.log

                def patched_logger_log(it: int, **kwargs):
                    locs = {
                        "it": it,
                        "loss_dict": kwargs.get("loss_dict", {}),
                        "collection_time": kwargs.get("collect_time", 0),
                        "learn_time": kwargs.get("learn_time", 0),
                        "rewbuffer": getattr(logger, "rewbuffer", []),
                        "lenbuffer": getattr(logger, "lenbuffer", []),
                        # ep_infos may arrive as a kwarg (rsl_rl >= 5.0) or sit on the logger
                        "ep_infos": (kwargs.get("ep_infos")
                                     or getattr(logger, "ep_infos", [])),
                        "extra_ep_info_buffer": getattr(logger, "extra_ep_info_buffer", []),
                    }
                    investigator._on_iteration(it, locs)
                    return original_log(it=it, **kwargs)

                logger.log = patched_logger_log
                try:
                    original_learn(num_learning_iterations, init_at_random_ep_len)
                finally:
                    logger.log = original_log
                    investigator._on_training_end()
            else:
                # rsl_rl < 5.0: patch runner.log directly
                original_log = runner.log

                @wraps(original_log)
                def patched_log(locs: dict, width: int = 80, pad: int = 35):
                    it = locs.get("it", 0)
                    investigator._on_iteration(it, locs)
                    return original_log(locs, width, pad)

                runner.log = patched_log
                try:
                    original_learn(num_learning_iterations, init_at_random_ep_len)
                finally:
                    runner.log = original_log
                    investigator._on_training_end()

        runner.learn = patched_learn
        self._installed = True
        print(f"[Investigator] Installed -> {self._log_dir}")

    # ---------------------------------------------------------
    # Fixed eval obs
    # ---------------------------------------------------------

    def _collect_fixed_eval_obs(self):
        env = self.runner.env
        n = self.cfg.n_eval_obs
        obs = self._get_actor_obs()

        if obs.shape[0] >= n:
            idx = torch.randperm(obs.shape[0])[:n]
            self._fixed_eval_obs = obs[idx].cpu()
        else:
            # Clamp to available envs to avoid duplicate-state bias
            warnings.warn(
                f"[Investigator] num_envs ({obs.shape[0]}) < n_eval_obs ({n}). "
                f"Using {obs.shape[0]} eval obs instead of {n}."
            )
            self._fixed_eval_obs = obs.cpu()

        torch.save(self._fixed_eval_obs, self._log_dir / "fixed_eval_obs.pt")
        print(f"[Investigator] Fixed eval obs: {self._fixed_eval_obs.shape}")

    # ---------------------------------------------------------
    # Iteration callback
    # ---------------------------------------------------------

    def _on_iteration(self, iteration: int, locs: dict | None = None):
        cfg = self.cfg
        # Mirror env-step count to RSL-RL's TensorBoard writer every iteration
        # so downstream consumers (e.g. Ray's MLflowLoggerCallback via
        # load_tensorboard_logs) can use it as the global step.
        total_steps = int(iteration) * int(self._step_scale)
        self._mirror_scalars_to_tensorboard(
            {"total_env_steps": float(total_steps)}, total_steps
        )
        if locs is not None:
            self._log_locs_scalars(iteration, locs)
        if iteration % cfg.log_interval == 0:
            self._compute_live_metrics(iteration)
        if iteration % cfg.weight_rank_interval == 0:
            self._compute_weight_ranks(iteration)
        if iteration % cfg.checkpoint_interval == 0:
            self._compute_checkpoint_metrics(iteration)
            self._dump_history()
        # Dump history at halfway point
        if (self._num_learning_iterations is not None and
            not self._half_dumped and
            iteration >= self._num_learning_iterations / 2):
            self._dump_history()
            self._half_dumped = True
            print(f"[Investigator] Halfway checkpoint dumped at iteration {iteration}")

    # ---------------------------------------------------------
    # Extra scalars scraped from rsl_rl's learn() locals
    # ---------------------------------------------------------

    def _log_locs_scalars(self, iteration: int, locs: dict):
        """Extract scalar metrics from rsl_rl's learn() locals -> history.json.

        - float/int: logged as-is.
        - deque/list/tuple of numbers: reduced with mean (skip if empty).
        - dict: flattened one level (e.g. loss_dict['value_loss'] ->
          investigator/train/value_loss).
        """
        prefix = "investigator/train/"
        # Map a couple of rsl_rl's buffer names to friendlier keys.
        rename = {"rewbuffer": "mean_reward", "lenbuffer": "mean_episode_length"}
        metrics = {}

        for key in self.cfg.locs_scalar_keys:
            if key not in locs:
                continue
            val = locs[key]
            out_key = rename.get(key, key)

            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    scalar = _to_scalar(sub_v)
                    if scalar is not None:
                        metrics[f"{prefix}{sub_k}"] = scalar
            else:
                seq = None
                if not isinstance(val, (str, bytes, torch.Tensor, np.generic)):
                    try:
                        seq = list(val)
                    except TypeError:
                        pass
                
                if seq is not None and len(seq) > 0 and isinstance(seq[0], dict):
                    gathered = {}
                    for d in seq:
                        for d_k, d_v in d.items():
                            gathered.setdefault(d_k, []).append(d_v)
                    for sub_k, sub_list in gathered.items():
                        scalar = _to_scalar(sub_list)
                        if scalar is not None:
                            metrics[f"{prefix}{sub_k}"] = scalar
                else:
                    scalar = _to_scalar(val)
                    if scalar is not None:
                        metrics[f"{prefix}{out_key}"] = scalar

        if metrics:
            self._log_metrics(metrics, iteration)

    def _on_training_end(self):
        # Save history locally (json + long-format csv for pandas)
        history_path = self._dump_history()

        backend_active = self._backend is not None and self._backend.is_active()

        # Final summary scalars
        if self.cfg.wandb_summary_on_end and backend_active:
            self._write_wandb_summary()

        # Upload entire investigator/ dir as artifact
        if self.cfg.wandb_artifact_on_end and backend_active:
            try:
                run_id = (wandb.run.id
                          if self.cfg.backend == "wandb"
                          and _WANDB_AVAILABLE and wandb.run is not None
                          else "run")
                self._backend.log_artifacts(self._log_dir, f"investigator-{run_id}")
            except Exception as e:
                warnings.warn(f"[Investigator] Artifact upload failed: {e}")

        print(f"[Investigator] Done. History -> {history_path}")

    def _dump_history(self) -> Path:
        """Write scalar history to disk without waiting for training to end.

        The JSON format includes a ``__version__`` key (currently ``2``) so that
        ``_load_history`` can distinguish files written with total-env-step
        x-values from legacy files that stored raw iteration counts.

        Note: saved ``.pt`` artefact files (grams, spectra, Jacobians) still use
        raw iteration numbers in their filenames for programmatic parsing.  The
        conversion to total env steps happens at display time via
        ``it * self._step_scale``.
        """
        history_path = self._log_dir / "history.json"
        if not self._history:
            return history_path

        serializable = {k: [(it, float(v)) for it, v in vals]
                        for k, vals in self._history.items()}
        serializable["__version__"] = 2
        serializable["__step_scale__"] = self._step_scale

        tmp_history_path = history_path.with_suffix(".json.tmp")
        with open(tmp_history_path, "w") as f:
            json.dump(serializable, f)
        tmp_history_path.replace(history_path)

        csv_path = self._log_dir / "history.csv"
        try:
            tmp_csv_path = csv_path.with_suffix(".csv.tmp")
            with open(tmp_csv_path, "w") as f:
                f.write("metric,total_env_steps,value\n")
                for k, vals in serializable.items():
                    if k.startswith("__"):
                        continue
                    for it, v in vals:
                        f.write(f"{k},{it},{v}\n")
            tmp_csv_path.replace(csv_path)
        except Exception as e:
            warnings.warn(f"[Investigator] history.csv dump failed: {e}")

        return history_path

    # [WANDB INTEGRATION] writes final/min/max/mean per metric to wandb.run.summary
    def _write_wandb_summary(self):
        summary = {}
        for key in [
            "investigator/actor/feature_erank",
            "investigator/actor/feature_erank_fixed",
            "investigator/critic/feature_erank",
            "investigator/gram/erank",
            "investigator/policy/jacobian_erank",
            "investigator/policy/jacobian_erank_sample_mean",
            "investigator/policy/jacobian_erank_sample_std",
            "investigator/actor/dead_neuron_frac",
            "investigator/critic/dead_neuron_frac",
            "investigator/actor/preact_norm",
            "investigator/policy/action_mean_variance",
        ]:
            vals = self._history.get(key, [])
            if vals:
                all_v = [v for _, v in vals]
                summary[f"{key}/final"] = all_v[-1]
                summary[f"{key}/min"] = min(all_v)
                summary[f"{key}/max"] = max(all_v)
                summary[f"{key}/mean"] = float(np.mean(all_v))

        dead = [v for _, v in self._history.get(
            "investigator/actor/dead_neuron_frac", [])]
        summary["investigator/collapse_detected"] = float(any(v > 0.5 for v in dead))
        self._backend.log_summary(summary)

    # ---------------------------------------------------------
    # Live metrics (every log_interval)
    # ---------------------------------------------------------

    def _compute_live_metrics(self, iteration: int):
        actor = self._get_actor()
        obs = self._get_actor_obs()
        metrics = {}

        with torch.no_grad():
            # -- Actor features --
            actor_feats = self._extract_penultimate_features(actor, obs)
            if actor_feats is not None:
                _, S_a, _ = torch.linalg.svd(actor_feats, full_matrices=False)
                metrics["investigator/actor/feature_erank"] = \
                    effective_rank(S_a, self.cfg.erank_eps)
                metrics["investigator/actor/pca_rank_99"] = pca_rank(S_a, 0.99)

                _S_norm = (S_a / S_a.sum()).cpu()
                for _i in range(min(20, len(_S_norm))):
                    metrics[f"investigator/actor/sv_{_i:02d}"] = _S_norm[_i].item()

                neuron_stds = actor_feats.std(dim=0)
                dead = (neuron_stds < self.cfg.dead_neuron_threshold).sum().item()
                total = actor_feats.shape[1]
                metrics["investigator/actor/dead_neurons"] = dead
                metrics["investigator/actor/dead_neuron_frac"] = dead / max(total, 1)
                metrics["investigator/actor/neuron_std_median"] = \
                    neuron_stds.median().item()

                # Backup raw arrays for post-hoc histogram reconstruction
                torch.save(S_a.cpu(), self._log_dir / f"spectra/actor_live_sv_{iteration}.pt")
                torch.save(neuron_stds.cpu(), self._log_dir / f"spectra/neuron_std_{iteration}.pt")

                # Live actor SV + neuron-std histograms
                if self.cfg.wandb_spectral_histograms and self._backend.is_active():
                    S_norm = (S_a / S_a.sum()).cpu().numpy()
                    self._backend.add_histogram(
                        metrics, "investigator/spectra/actor_features",
                        S_norm, iteration)
                    self._backend.add_histogram(
                        metrics, "investigator/actor/neuron_std_dist",
                        neuron_stds.cpu().numpy(), iteration)

            # -- Pre-activation norm --
            preact = self._compute_preactivation_norm(actor, obs)
            if preact is not None:
                metrics["investigator/actor/preact_norm"] = preact[0]
                metrics["investigator/actor/preact_norm_std"] = preact[1]

            # -- Policy variance + update magnitude --
            action_means = self._get_action_means(actor, obs)
            if action_means is not None:
                action_var_per_dim = action_means.var(dim=0)
                metrics["investigator/policy/action_mean_variance"] = \
                    action_var_per_dim.mean().item()
                metrics["investigator/policy/action_mean_variance_std"] = \
                    action_var_per_dim.std().item()

                action_means_cpu = action_means.detach().cpu()
                if (self._prev_eval_actions is not None
                        and action_means_cpu.shape == self._prev_eval_actions.shape):
                    metrics["investigator/policy/update_magnitude"] = (
                        (action_means_cpu - self._prev_eval_actions).abs().mean().item()
                    )
                self._prev_eval_actions = action_means_cpu

            # -- Critic features --
            critic = self._get_critic()
            critic_feats = self._extract_penultimate_features(critic, obs)
            if critic_feats is not None:
                _, S_c, _ = torch.linalg.svd(critic_feats, full_matrices=False)
                metrics["investigator/critic/feature_erank"] = \
                    effective_rank(S_c, self.cfg.erank_eps)
                metrics["investigator/critic/pca_rank_99"] = pca_rank(S_c, 0.99)
                neuron_stds_c = critic_feats.std(dim=0)
                dead_c = (neuron_stds_c < self.cfg.dead_neuron_threshold).sum().item()
                total_c = critic_feats.shape[1]
                metrics["investigator/critic/dead_neurons"] = dead_c
                metrics["investigator/critic/dead_neuron_frac"] = dead_c / max(total_c, 1)

        self._log_metrics(metrics, iteration)

    # ---------------------------------------------------------
    # Weight ranks (every weight_rank_interval)
    # ---------------------------------------------------------

    def _compute_weight_ranks(self, iteration: int):
        metrics = {}
        layer_eranks_actor = []

        with torch.no_grad():
            for label, model in [("actor", self._get_actor())]:
                for i, (name, layer) in enumerate(_find_linear_layers(model)):
                    W = layer.weight
                    _, S, _ = torch.linalg.svd(W, full_matrices=False)
                    er = effective_rank(S, self.cfg.erank_eps)
                    metrics[f"investigator/weights/{label}/layer_{i}_erank"] = er

                    # Backup raw SVs for post-hoc histogram reconstruction
                    torch.save(S.cpu(), self._log_dir / f"spectra/weight_{label}_L{i}_sv_{iteration}.pt")

                    if label == "actor":
                        layer_eranks_actor.append((f"L{i}", er))

                    # Per-layer weight SV histogram
                    if self.cfg.wandb_spectral_histograms and self._backend.is_active():
                        self._backend.add_histogram(
                            metrics, f"investigator/spectra/weight_{label}_L{i}",
                            S.cpu().numpy(), iteration)

                    # Per-factor spectra for LinOP layers (Huh et al. Fig 14)
                    if hasattr(layer, "factors"):
                        for j, factor in enumerate(layer.factors):
                            _, S_f, _ = torch.linalg.svd(
                                factor.detach(), full_matrices=False)
                            metrics[f"investigator/weights/{label}/layer_{i}_factor_{j}_erank"] = (
                                effective_rank(S_f, self.cfg.erank_eps)
                            )
                            if self.cfg.save_spectra:
                                torch.save(
                                    S_f.cpu(),
                                    self._log_dir / f"spectra/weight_{label}_L{i}_F{j}_sv_{iteration}.pt")

        self._log_metrics(metrics, iteration)

    # ---------------------------------------------------------
    # Checkpoint metrics (expensive, every checkpoint_interval)
    # ---------------------------------------------------------

    def _compute_checkpoint_metrics(self, iteration: int):
        actor = self._get_actor()
        eval_obs = self._fixed_eval_obs.to(self.device)
        metrics = {}

        # -- Gram matrix + feature erank on fixed eval obs --
        with torch.no_grad():
            features = self._extract_penultimate_features(actor, eval_obs)
            if features is not None:
                # Feature erank on fixed eval obs
                _, S_fixed, V_fixed = torch.linalg.svd(
                    features.cpu(), full_matrices=False)
                feat_er_fixed = effective_rank(S_fixed, self.cfg.erank_eps)
                metrics["investigator/actor/feature_erank_fixed"] = feat_er_fixed

                gram = compute_gram_cosine(features)
                _, S_gram, _ = torch.linalg.svd(gram)
                gram_er = effective_rank(S_gram, self.cfg.erank_eps)
                metrics["investigator/gram/erank"] = gram_er

                # Local save
                if self.cfg.save_grams:
                    torch.save(gram.cpu(),
                               self._log_dir / f"grams/gram_{iteration}.pt")
                if self.cfg.save_features:
                    torch.save(features.cpu(),
                               self._log_dir / f"features/features_{iteration}.pt")

                # Gram SV histogram
                if self.cfg.wandb_spectral_histograms and self._backend.is_active():
                    S_n = (S_gram / S_gram.sum()).cpu().numpy()
                    self._backend.add_histogram(
                        metrics, "investigator/spectra/gram_sv", S_n, iteration)

                # Gram heatmap image (yellow — default)
                if self.cfg.wandb_gram_image and self._backend.is_active():
                    fig = self._render_gram_figure(
                        gram.cpu(), iteration, gram_er)
                    if fig is not None:
                        self._backend.add_image(
                            metrics, "investigator/gram/heatmap", fig, iteration)
                        plt.close(fig)

                # Gram heatmap image (blue variant)
                if self.cfg.wandb_gram_image and self._backend.is_active():
                    fig_blue = self._render_gram_figure(
                        gram.cpu(), iteration, gram_er,
                        cmap_override=_CMAP_WB)
                    if fig_blue is not None:
                        self._backend.add_image(
                            metrics, "investigator/gram/heatmap_blue",
                            fig_blue, iteration)
                        plt.close(fig_blue)

                # Feature activation heatmap image
                if self.cfg.wandb_feature_heatmap and self._backend.is_active():
                    fig = self._render_feature_heatmap(
                        features.cpu(), V_fixed, iteration, feat_er_fixed)
                    if fig is not None:
                        self._backend.add_image(
                            metrics, "investigator/features/heatmap",
                            fig, iteration)
                        plt.close(fig)

        # -- Jacobian rank --
        jac_obs = eval_obs[:self.cfg.jacobian_batch_size]
        jac_er, jac_S, jac_matrix, jac_std, jac_ps_eranks = \
            self._compute_jacobian_erank(actor, jac_obs)
        if jac_er is not None:
            metrics["investigator/policy/jacobian_erank"] = jac_er

            # Per-sample Jacobian erank stats
            if jac_ps_eranks is not None:
                metrics["investigator/policy/jacobian_erank_sample_mean"] = \
                    jac_ps_eranks.mean().item()
                metrics["investigator/policy/jacobian_erank_sample_std"] = \
                    jac_ps_eranks.std().item()
                # Per-sample Jacobian erank histogram
                if self.cfg.wandb_spectral_histograms and self._backend.is_active():
                    self._backend.add_histogram(
                        metrics, "investigator/spectra/jacobian_persample_eranks",
                        jac_ps_eranks.numpy(), iteration)

            if self.cfg.save_jacobian_spectra and jac_S is not None:
                torch.save(jac_S.cpu(),
                           self._log_dir / f"jacobians/jac_sv_{iteration}.pt")
            if self.cfg.save_jacobian_spectra and jac_ps_eranks is not None:
                torch.save(jac_ps_eranks,
                           self._log_dir / f"jacobians/jac_persample_eranks_{iteration}.pt")
            if self.cfg.save_jacobians and jac_matrix is not None:
                torch.save(jac_matrix.cpu(),
                           self._log_dir / f"jacobians/jac_matrix_{iteration}.pt")
            if self.cfg.save_jacobians and jac_std is not None:
                torch.save(jac_std.cpu(),
                           self._log_dir / f"jacobians/jac_std_{iteration}.pt")

            # Jacobian mean SV histogram
            if (self.cfg.wandb_spectral_histograms and self._backend.is_active()
                    and jac_S is not None):
                self._backend.add_histogram(
                    metrics, "investigator/spectra/jacobian_sv",
                    jac_S.cpu().numpy(), iteration)

            # Jacobian mean heatmap image
            if (self.cfg.wandb_jacobian_heatmap and self._backend.is_active()
                    and jac_matrix is not None):
                fig = self._render_jacobian_heatmap(
                    jac_matrix.cpu(), iteration, jac_er)
                if fig is not None:
                    self._backend.add_image(
                        metrics, "investigator/jacobian/heatmap", fig, iteration)
                    plt.close(fig)

            # Jacobian std heatmap image
            if (self.cfg.wandb_jacobian_heatmap and self._backend.is_active()
                    and jac_std is not None):
                fig = self._render_jacobian_std_heatmap(
                    jac_std.cpu(), iteration)
                if fig is not None:
                    self._backend.add_image(
                        metrics, "investigator/jacobian/std_heatmap",
                        fig, iteration)
                    plt.close(fig)

        # -- Phase-conditioned Jacobian --
        if self._phase_fn is not None and jac_er is not None:
            try:
                phases = self._phase_fn(self.runner.env)[:self.cfg.jacobian_batch_size]
                phases = phases.to(jac_obs.device)
                for p in phases.unique():
                    mask = phases == p
                    if mask.sum() >= 16:
                        er_p, _, _, _, _ = self._compute_jacobian_erank(
                            actor, jac_obs[mask])
                        if er_p is not None:
                            metrics[f"investigator/policy/jacobian_erank_phase{p.item()}"] = er_p
            except Exception as e:
                warnings.warn(f"[Investigator] Phase Jacobian failed: {e}")

        # Per-layer weight heatmap image
        if self.cfg.wandb_weight_heatmap and self._backend.is_active():
            fig = self._render_weight_heatmap(actor, iteration)
            if fig is not None:
                self._backend.add_image(
                    metrics, "investigator/weights/heatmap", fig, iteration)
                plt.close(fig)

        # -- Full weight spectra (disk only) --
        if self.cfg.save_spectra:
            self._save_weight_spectra(actor, iteration, "actor")
            # Backup raw weight matrices for post-hoc weight-heatmap reconstruction
            weights = {f"layer_{i}_{name}": layer.weight.cpu()
                       for i, (name, layer) in enumerate(_find_linear_layers(actor))}
            torch.save(weights, self._log_dir / f"spectra/actor_weights_{iteration}.pt")

        self._log_metrics(metrics, iteration)

    # ---------------------------------------------------------
    # Feature extraction via hooks
    # ---------------------------------------------------------

    def _extract_penultimate_features(
        self, model: nn.Module, obs: torch.Tensor
    ) -> torch.Tensor | None:
        captured = {}
        inner = getattr(model, "mlp", model)
        try:
            _, penult = _find_penultimate_linear(inner)
        except ValueError:
            return None

        def hook(module, inp, out):
            captured["f"] = out.detach()

        h = penult.register_forward_hook(hook)
        try:
            with torch.no_grad():
                inner(obs)
        finally:
            h.remove()
        return captured.get("f")

    def _compute_preactivation_norm(
        self, model: nn.Module, obs: torch.Tensor
    ) -> tuple[float, float] | None:
        captured = {}
        inner = getattr(model, "mlp", model)
        try:
            _, penult = _find_penultimate_linear(inner)
        except ValueError:
            return None

        def hook(module, inp, out):
            captured["p"] = out.detach()

        h = penult.register_forward_hook(hook)
        try:
            with torch.no_grad():
                inner(obs)
        finally:
            h.remove()

        p = captured.get("p")
        if p is None:
            return None
        norms = p.norm(dim=-1)
        return norms.mean().item(), norms.std().item()

    def _get_action_means(
        self, actor: nn.Module, obs: torch.Tensor
    ) -> torch.Tensor | None:
        """Get action means from the actor (nn.Sequential -> raw tensor)."""
        inner = getattr(actor, "mlp", actor)
        with torch.no_grad():
            try:
                return inner(obs).detach()
            except Exception:
                return None

    # ---------------------------------------------------------
    # Jacobian
    # ---------------------------------------------------------

    def _compute_jacobian_erank(
        self, actor: nn.Module, obs: torch.Tensor
    ) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        try:
            if self.cfg.jacobian_use_vmap and hasattr(torch, "func"):
                return self._jacobian_vmap(actor, obs)
            return self._jacobian_loop(actor, obs)
        except Exception as e:
            warnings.warn(f"[Investigator] Jacobian failed: {e}")
            return None, None, None, None, None

    def _jacobian_vmap(self, actor, obs):
        from torch.func import jacrev, vmap

        inner = getattr(actor, "mlp", actor)

        def fwd(x):
            return inner(x.unsqueeze(0)).squeeze(0)

        inner.eval()
        with torch.enable_grad():
            J = vmap(jacrev(fwd))(obs)  # [B, a_dim, o_dim]
        inner.train()

        J_mean = J.mean(dim=0)
        J_std = J.std(dim=0)
        _, S, _ = torch.linalg.svd(J_mean, full_matrices=False)
        per_sample_eranks = torch.tensor([
            effective_rank(torch.linalg.svdvals(J[i]), self.cfg.erank_eps)
            for i in range(J.shape[0])
        ])
        return (effective_rank(S, self.cfg.erank_eps), S.detach().cpu(),
                J_mean.detach().cpu(), J_std.detach().cpu(), per_sample_eranks)

    def _jacobian_loop(self, actor, obs):
        inner = getattr(actor, "mlp", actor)
        obs_g = obs.detach().requires_grad_(True)
        inner.eval()
        out = inner(obs_g)
        a_dim = out.shape[-1]

        J = torch.zeros(obs.shape[0], a_dim, obs.shape[-1], device=obs.device)
        for a in range(a_dim):
            g = torch.zeros_like(out)
            g[:, a] = 1.0
            grads = torch.autograd.grad(
                out, obs_g, g, retain_graph=(a < a_dim - 1))[0]
            J[:, a, :] = grads

        inner.train()
        J_mean = J.mean(dim=0)
        J_std = J.std(dim=0)
        _, S, _ = torch.linalg.svd(J_mean, full_matrices=False)
        per_sample_eranks = torch.tensor([
            effective_rank(torch.linalg.svdvals(J[i]), self.cfg.erank_eps)
            for i in range(J.shape[0])
        ])
        return (effective_rank(S, self.cfg.erank_eps), S.detach().cpu(),
                J_mean.detach().cpu(), J_std.detach().cpu(), per_sample_eranks)

    # ---------------------------------------------------------
    # Weight spectra (disk)
    # ---------------------------------------------------------

    def _save_weight_spectra(self, model, iteration, label):
        spectra = {}
        with torch.no_grad():
            for i, (name, layer) in enumerate(_find_linear_layers(model)):
                W = layer.weight
                _, S, _ = torch.linalg.svd(W, full_matrices=False)
                spectra[f"layer_{i}_{name}"] = S.cpu()
        torch.save(
            spectra,
            self._log_dir / f"spectra/{label}_spectra_{iteration}.pt"
        )

    # ---------------------------------------------------------
    # Model access (rsl_rl >= 5.0: alg.actor / alg.critic directly)
    # ---------------------------------------------------------

    def _get_actor_obs(self) -> torch.Tensor:
        """Return a flat observation tensor for the actor (rsl_rl >= 5.0.0 compatible)."""
        obs_td = self.runner.env.get_observations().to(self.device)
        actor = self.runner.alg.actor
        if hasattr(actor, "obs_groups"):
            return torch.cat([obs_td[g] for g in actor.obs_groups], dim=-1)
        try:
            return obs_td["policy"]
        except (KeyError, TypeError):
            return next(iter(obs_td.values()))

    def _get_actor(self) -> nn.Module:
        # Return the inner MLP so it accepts flat tensors (needed for hooks / Jacobian)
        actor = self.runner.alg.actor
        return getattr(actor, "mlp", actor)

    def _get_critic(self) -> nn.Module:
        critic = self.runner.alg.critic
        return getattr(critic, "mlp", critic)

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    def _get_scalar_writer(self):
        if self._scalar_writer is not None:
            return self._scalar_writer

        for owner in (self.runner, getattr(self.runner, "logger", None)):
            if owner is None:
                continue
            if hasattr(owner, "add_scalar"):
                self._scalar_writer = owner
                return self._scalar_writer
            for attr in (
                "writer",
                "summary_writer",
                "tb_writer",
                "tensorboard_writer",
            ):
                writer = getattr(owner, attr, None)
                if hasattr(writer, "add_scalar"):
                    self._scalar_writer = writer
                    return self._scalar_writer
        return None

    def _mirror_scalars_to_tensorboard(self, scalars: dict[str, float],
                                       iteration: int):
        writer = self._get_scalar_writer()
        if writer is None:
            return
        try:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, iteration)
            if hasattr(writer, "flush"):
                writer.flush()
        except Exception as e:
            warnings.warn(f"[Investigator] TensorBoard scalar mirror failed: {e}")

    def _log_metrics(self, metrics: dict, iteration: int):
        scalars, media = _split_metric_payload(metrics)
        total_steps = iteration * self._step_scale

        # History tracking is backend-agnostic; non-scalars are skipped.
        for k, v in scalars.items():
            self._history.setdefault(k, []).append((total_steps, v))

        if scalars:
            self._mirror_scalars_to_tensorboard(scalars, total_steps)

        if self.cfg.wandb_log and self._backend is not None \
                and self._backend.is_active():
            self._backend.flush(scalars, media, total_steps)

    # ---------------------------------------------------------
    # PPO trust-region (optional — call from patched PPO.update)
    # ---------------------------------------------------------

    def log_ppo_ratios(self, ratios: torch.Tensor, clip_param: float,
                       iteration: int):
        with torch.no_grad():
            above = ratios[ratios > 1 + clip_param]
            below = ratios[ratios < 1 - clip_param]
            frac = (
                (ratios > 1 + clip_param).float().mean()
                + (ratios < 1 - clip_param).float().mean()
            ).item()
            metrics = {
                "investigator/ppo/frac_clipped": frac,
                "investigator/ppo/avg_ratio_above":
                    above.mean().item() if len(above) else 1 + clip_param,
                "investigator/ppo/avg_ratio_below":
                    below.mean().item() if len(below) else 1 - clip_param,
                "investigator/ppo/ratio_mean": ratios.mean().item(),
                "investigator/ppo/ratio_std": ratios.std().item(),
            }
            # PPO ratio distribution histogram
            if self.cfg.wandb_spectral_histograms and self._backend is not None \
                    and self._backend.is_active():
                self._backend.add_histogram(
                    metrics, "investigator/ppo/ratio_distribution",
                    ratios.cpu().numpy(), iteration)
                # Backup raw ratios for post-hoc histogram reconstruction
                torch.save(ratios.detach().cpu(),
                           self._log_dir / f"ppo/ratios_{iteration}.pt")

        self._log_metrics(metrics, iteration)

    # ===================================================================
    # Gram matrix rendering
    # ===================================================================

    def _render_gram_figure(self, gram: torch.Tensor, iteration: int,
                            erank_val: float,
                            cmap_override=None):

        gram_np = gram.numpy()
        if self._sort_indices is not None:
            idx = self._sort_indices.numpy()
            gram_np = gram_np[np.ix_(idx, idx)]

        fig, ax = plt.subplots(
            1, 1,
            figsize=self.cfg.figsize_gram,
        )

        # Full Gram panel (kept for future reuse)
        # norm, cmap = _gram_colornorm(gram_np)
        # im0 = axes[0].imshow(gram_np, cmap=cmap, norm=norm,
        #                      aspect="equal", interpolation="nearest")
        # axes[0].set_title(
        #     f"Gram (full) -- iter {iteration}  |  erank = {erank_val:.1f}",
        #     fontsize=8)
        # axes[0].set_xlabel("obs index")
        # axes[0].set_ylabel("obs index")
        # div0 = make_axes_locatable(axes[0])
        # fig.colorbar(im0, cax=div0.append_axes("right", size="4%", pad=0.05))

        # Pattern view (Hu et al. style) — cmap_override switches colormap
        gp, norm_p, cmap_p, vmin_p, vmax_p = _gram_pattern(
            gram_np, cmap=cmap_override)
        im1 = ax.imshow(gp, cmap=cmap_p, norm=norm_p,
                        vmin=vmin_p, vmax=vmax_p,
                        aspect="equal", interpolation="nearest")
        ax.set_title(
            f"Gram (pattern) -- {_fmt_total_steps(iteration * self._step_scale)}", fontsize=8)
        ax.set_xlabel("obs index")
        ax.set_ylabel("obs index")
        div1 = make_axes_locatable(ax)
        fig.colorbar(im1, cax=div1.append_axes("right", size="4%", pad=0.05))
        fig.tight_layout()
        return fig

    # ===================================================================
    # Feature activation heatmap (N x D)
    # ===================================================================

    def _render_feature_heatmap(self, features: torch.Tensor,
                                V: torch.Tensor, iteration: int,
                                erank_val: float):
        """Render N x D feature activation heatmap.

        Rows sorted by sort_fn (same as Gram), columns sorted by
        participation in the top right-singular-vector.
        """

        feat_np = features.numpy()
        N, D = feat_np.shape

        # Sort rows (observations) by sort_fn indices
        if self._sort_indices is not None:
            feat_np = feat_np[self._sort_indices.numpy()]

        # Sort columns by top right-singular-vector magnitude
        col_order = np.argsort(-np.abs(V[0].numpy()))
        feat_np = feat_np[:, col_order]

        w, h = 10.0, 8.0
        fig, ax = plt.subplots(figsize=(w, h))
        vabs = max(abs(feat_np.min()), abs(feat_np.max()))
        if vabs > 0:
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        else:
            norm = None
        im = ax.imshow(feat_np, cmap=_CMAP_BWY, norm=norm,
                   aspect="auto", interpolation="nearest")
        ax.set_title(f"Feature Activations -- {_fmt_total_steps(iteration * self._step_scale)}  |  "
                     f"erank = {erank_val:.1f}", fontsize=9)
        ax.set_xlabel("neuron (sorted by top SV)")
        ax.set_ylabel("obs index")
        div = make_axes_locatable(ax)
        fig.colorbar(im, cax=div.append_axes("right", size="3%", pad=0.05))
        fig.tight_layout()
        return fig

    # ===================================================================
    # Jacobian heatmap (action_dim x obs_dim)
    # ===================================================================

    def _render_jacobian_heatmap(self, J_mean: torch.Tensor, iteration: int,
                                 erank_val: float):
        """Render action_dim x obs_dim Jacobian as a semantically labeled heatmap."""

        J_np = J_mean.numpy()
        a_dim, o_dim = J_np.shape

        # Row labels (actions)
        row_labels = ACTION_LABELS[:a_dim] if a_dim <= len(ACTION_LABELS) \
            else [f"a{i}" for i in range(a_dim)]

        # Figure: natural aspect ratio (short & wide for 12 x 49)
        w = max(6.5, o_dim / 6)
        h = max(2.5, a_dim / 3)
        fig, ax = plt.subplots(figsize=(w, h))

        vabs = max(abs(J_np.min()), abs(J_np.max()))
        if vabs > 0:
            norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        else:
            norm = None
        im = ax.imshow(J_np, cmap=_CMAP_BWY, norm=norm, aspect="auto",
                       interpolation="nearest")

        ax.set_yticks(range(a_dim))
        ax.set_yticklabels(row_labels, fontsize=7)

        # Obs group separators and labels
        group_ticks = []
        for gname, g_start, g_end in OBS_GROUPS:
            if g_start < o_dim:
                mid = (g_start + min(g_end, o_dim)) / 2
                group_ticks.append((mid, gname))
                if g_start > 0:
                    ax.axvline(g_start - 0.5, color="k",
                               linewidth=0.5, alpha=0.4)
        if group_ticks:
            ax.set_xticks([t for t, _ in group_ticks])
            ax.set_xticklabels([l for _, l in group_ticks],
                               fontsize=6, rotation=30, ha="right")

        # Leg group separators (every 3 actions)
        for i in range(3, a_dim, 3):
            ax.axhline(i - 0.5, color="k", linewidth=0.5, alpha=0.4)

        ax.set_title(f"Jacobian -- {_fmt_total_steps(iteration * self._step_scale)}  |  "
                     f"erank = {erank_val:.1f}", fontsize=9)
        div = make_axes_locatable(ax)
        fig.colorbar(im, cax=div.append_axes("right", size="3%", pad=0.05))
        fig.tight_layout()
        return fig

    # ===================================================================
    # Jacobian std heatmap (action_dim x obs_dim)
    # ===================================================================

    def _render_jacobian_std_heatmap(self, J_std: torch.Tensor, iteration: int):
        """Render element-wise std of J across samples as a heatmap."""

        J_np = J_std.numpy()
        a_dim, o_dim = J_np.shape

        row_labels = ACTION_LABELS[:a_dim] if a_dim <= len(ACTION_LABELS) \
            else [f"a{i}" for i in range(a_dim)]

        w = max(6.5, o_dim / 6)
        h = max(2.5, a_dim / 3)
        fig, ax = plt.subplots(figsize=(w, h))

        vmax = J_np.max() if J_np.max() > 0 else 1.0
        im = ax.imshow(J_np, cmap="magma", vmin=0.0, vmax=vmax,
                       aspect="auto", interpolation="nearest")

        ax.set_yticks(range(a_dim))
        ax.set_yticklabels(row_labels, fontsize=7)

        group_ticks = []
        for gname, g_start, g_end in OBS_GROUPS:
            if g_start < o_dim:
                mid = (g_start + min(g_end, o_dim)) / 2
                group_ticks.append((mid, gname))
                if g_start > 0:
                    ax.axvline(g_start - 0.5, color="w",
                               linewidth=0.5, alpha=0.4)
        if group_ticks:
            ax.set_xticks([t for t, _ in group_ticks])
            ax.set_xticklabels([l for _, l in group_ticks],
                               fontsize=6, rotation=30, ha="right")

        for i in range(3, a_dim, 3):
            ax.axhline(i - 0.5, color="w", linewidth=0.5, alpha=0.4)

        ax.set_title(f"Jacobian Std -- {_fmt_total_steps(iteration * self._step_scale)}", fontsize=9)
        div = make_axes_locatable(ax)
        fig.colorbar(im, cax=div.append_axes("right", size="3%", pad=0.05))
        fig.tight_layout()
        return fig

    # ===================================================================
    # Weight heatmap (per-layer)
    # ===================================================================

    def _render_weight_heatmap(self, model: nn.Module, iteration: int):
        """Render weight matrices of all Linear layers as heatmap subplots."""

        layers = _find_linear_layers(model)
        if not layers:
            return None

        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 3.0), squeeze=False)

        with torch.no_grad():
            for i, (name, layer) in enumerate(layers):
                W_t = layer.weight
                W_np = W_t.cpu().numpy()
                _, S, _ = torch.linalg.svd(W_t, full_matrices=False)
                er = effective_rank(S, self.cfg.erank_eps)

                ax = axes[0, i]
                vabs = max(abs(W_np.min()), abs(W_np.max()))
                if vabs > 0:
                    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
                else:
                    norm = None
                im = ax.imshow(W_np, cmap=_CMAP_BWY, norm=norm, aspect="auto",
                               interpolation="nearest")
                ax.set_title(f"L{i} ({W_np.shape[0]}x{W_np.shape[1]})\n"
                             f"er={er:.1f}", fontsize=7)
                ax.tick_params(labelsize=5)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Weight Matrices -- {_fmt_total_steps(iteration * self._step_scale)}", fontsize=9)
        fig.tight_layout()
        return fig

    # ===================================================================
    # Post-hoc report generation
    # ===================================================================

    def generate_report(self, compare_dirs: list[str | Path] | None = None):
        """Generate matplotlib plots from saved data.

        Each plot is wrapped in try/except so a failure in one (e.g. drifted
        layer names, missing files, matplotlib weirdness) doesn't kill the
        whole report — and, critically, doesn't propagate up to crash the
        training subprocess and mark the trial FAILED in Ray/MLflow.  The
        training itself has already finished by the time we get here; the
        worst a report bug should do is print a warning.
        """

        plots_dir = self._log_dir / "plots"
        history = self._load_history(self._log_dir)
        all_h = [("this_run", history)]
        if compare_dirs:
            for d in compare_dirs:
                all_h.append((Path(d).parent.name,
                              self._load_history(Path(d))))

        for fn, args in (
            (self._plot_rank_evolution,        (all_h, plots_dir)),
            (self._plot_gram_evolution_grid,   (plots_dir,)),
            (self._plot_feature_heatmap_grid,  (plots_dir,)),
            (self._plot_jacobian_heatmap_grid, (plots_dir,)),
            (self._plot_spectral_evolution,    (plots_dir,)),
            (self._plot_collapse_indicators,   (all_h, plots_dir)),
        ):
            try:
                fn(*args)
            except Exception as e:
                warnings.warn(f"[Investigator] {fn.__name__} failed: {e}")

        print(f"[Investigator] Report -> {plots_dir}")

    def _load_history(self, inv_dir: Path) -> dict:
        p = inv_dir / "history.json"
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            version = raw.pop("__version__", 1)
            raw.pop("__step_scale__", None)
            if version < 2:
                warnings.warn(
                    f"[Investigator] {p} uses legacy format (raw iterations). "
                    f"X-axis values will not reflect total env steps. "
                    f"Re-run training to generate a v2 history file."
                )
            return raw
        return self._history

    def _plot_rank_evolution(self, all_h, out):
        keys = [
            ("investigator/actor/feature_erank", "Actor Feature erank"),
            ("investigator/policy/jacobian_erank", "Jacobian erank"),
            ("investigator/gram/erank", "Gram erank"),
        ]
        fig, axes = plt.subplots(1, len(keys),
                                 figsize=self.cfg.figsize_evolution)
        for ax, (k, t) in zip(axes, keys):
            all_its = []
            for label, h in all_h:
                if k in h:
                    its, vs = zip(*h[k])
                    ax.plot(its, vs, label=label, linewidth=1.5)
                    all_its.extend(its)
            ax.set_title(t, fontsize=9)
            _setup_sci_xaxis(ax, all_its)
            ax.set_ylabel("effective rank")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Rank Evolution", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"rank_evolution.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_gram_evolution_grid(self, out):
        gdir = self._log_dir / "grams"
        gfiles = sorted(gdir.glob("gram_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
        if not gfiles:
            return
        n = min(6, len(gfiles))
        idxs = np.linspace(0, len(gfiles) - 1, n, dtype=int)
        sel = [gfiles[i] for i in idxs]

        fig, axes = plt.subplots(
            3, n, figsize=(n * 3.2, 10), height_ratios=[3, 3, 1],
            gridspec_kw={"hspace": .35, "wspace": .3})
        if n == 1:
            axes = axes.reshape(3, 1)

        for col, gf in enumerate(sel):
            gram = torch.load(gf, weights_only=True)
            g_np = gram.numpy()
            it = int(gf.stem.split("_")[1])
            if self._sort_indices is not None:
                idx = self._sort_indices.numpy()
                g_np = g_np[np.ix_(idx, idx)]

            _, Sg, _ = torch.linalg.svd(gram)
            er = effective_rank(Sg, self.cfg.erank_eps)

            # Row 0: 3-color diagnostic
            ax_g = axes[0, col]
            norm, cmap = _gram_colornorm(g_np)
            ax_g.imshow(g_np, cmap=cmap, norm=norm,
                        aspect="equal", interpolation="nearest")
            ax_g.set_title(f"{_fmt_total_steps(it * self._step_scale)}\ner={er:.1f}", fontsize=8)
            ax_g.tick_params(labelsize=6)

            # Row 1: pattern view (white->yellow)
            ax_p = axes[1, col]
            gp, norm_p, cmap_p, vmin_p, vmax_p = _gram_pattern(g_np)
            ax_p.imshow(gp, cmap=cmap_p, norm=norm_p,
                        vmin=vmin_p, vmax=vmax_p,
                        aspect="equal", interpolation="nearest")
            ax_p.set_title("pattern", fontsize=7)
            ax_p.tick_params(labelsize=6)

            # Row 2: spectral decay
            ax_s = axes[2, col]
            Sn = (Sg / Sg.sum()).numpy()
            ax_s.semilogy(Sn, color="#2E86AB", linewidth=1.0)
            ax_s.axhline(1 / len(Sn), color="#A23B72",
                         linestyle="--", linewidth=.6)
            ax_s.tick_params(labelsize=6)
            if col == 0:
                ax_s.set_ylabel("normalized sigma", fontsize=7)

        fig.suptitle("Gram Matrix Evolution", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"gram_evolution.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_feature_heatmap_grid(self, out):
        fdir = self._log_dir / "features"
        ffiles = sorted(fdir.glob("features_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
        if not ffiles:
            return
        n = min(4, len(ffiles))
        idxs = np.linspace(0, len(ffiles) - 1, n, dtype=int)
        sel = [ffiles[i] for i in idxs]

        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4), squeeze=False)
        for col, ff in enumerate(sel):
            features = torch.load(ff, weights_only=True)
            it = int(ff.stem.split("_")[1])
            feat_np = features.numpy()

            if self._sort_indices is not None:
                feat_np = feat_np[self._sort_indices.numpy()]

            _, _, V = torch.linalg.svd(features, full_matrices=False)
            col_order = np.argsort(-np.abs(V[0].numpy()))
            feat_np = feat_np[:, col_order]

            _, S_f, _ = torch.linalg.svd(features, full_matrices=False)
            er = effective_rank(S_f, self.cfg.erank_eps)

            ax = axes[0, col]
            ax.imshow(feat_np, cmap="viridis", aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"{_fmt_total_steps(it * self._step_scale)}\ner={er:.1f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel("obs index", fontsize=7)

        fig.suptitle("Feature Activation Evolution", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"feature_evolution.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_jacobian_heatmap_grid(self, out):
        jdir = self._log_dir / "jacobians"
        jfiles = sorted(jdir.glob("jac_matrix_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
        if not jfiles:
            return
        n = min(4, len(jfiles))
        idxs = np.linspace(0, len(jfiles) - 1, n, dtype=int)
        sel = [jfiles[i] for i in idxs]

        fig, axes = plt.subplots(1, n, figsize=(n * 5, 3.0), squeeze=False)
        for col, jf in enumerate(sel):
            J_mean = torch.load(jf, weights_only=True)
            it = int(jf.stem.split("_")[-1])
            J_np = J_mean.numpy()
            a_dim, o_dim = J_np.shape

            _, S_j, _ = torch.linalg.svd(J_mean, full_matrices=False)
            er = effective_rank(S_j, self.cfg.erank_eps)

            ax = axes[0, col]
            vabs = max(abs(J_np.min()), abs(J_np.max()))
            if vabs > 0:
                norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            else:
                norm = None
            ax.imshow(J_np, cmap="RdBu_r", norm=norm, aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"{_fmt_total_steps(it * self._step_scale)}\ner={er:.1f}", fontsize=8)

            row_labels = ACTION_LABELS[:a_dim] if a_dim <= len(ACTION_LABELS) \
                else [f"a{i}" for i in range(a_dim)]
            ax.set_yticks(range(a_dim))
            ax.set_yticklabels(row_labels, fontsize=5)
            ax.tick_params(axis="x", labelsize=5)

            # Obs group separators
            for _, g_start, _ in OBS_GROUPS:
                if 0 < g_start < o_dim:
                    ax.axvline(g_start - 0.5, color="k",
                               linewidth=0.4, alpha=0.3)
            for i in range(3, a_dim, 3):
                ax.axhline(i - 0.5, color="k", linewidth=0.4, alpha=0.3)

        fig.suptitle("Jacobian Evolution", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"jacobian_evolution.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_spectral_evolution(self, out):
        sdir = self._log_dir / "spectra"
        sfiles = sorted(sdir.glob("actor_spectra_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
        if not sfiles:
            return
        n = min(4, len(sfiles))
        idxs = np.linspace(0, len(sfiles) - 1, n, dtype=int)
        sel = [sfiles[i] for i in idxs]

        # Build lnames from the *union* of layers across all selected
        # snapshots — not just the first.  For some architectures (SimBa
        # in particular) the per-snapshot layer set drifts over training
        # (e.g. layer_0_embed appears only in some checkpoints).  Indexing
        # blindly with the first-snapshot's keys would raise KeyError and
        # kill the trial AFTER training has already finished — see the
        # 'anymal_flat_v1' anymal/simba runs at iter ~290 / 6:03 elapsed.
        snapshots = [torch.load(sf, weights_only=True) for sf in sel]
        seen: dict[str, None] = {}
        for sp in snapshots:
            for k in sp.keys():
                seen.setdefault(k, None)
        lnames = list(seen.keys())
        nl = len(lnames)

        fig, axes = plt.subplots(nl, n, figsize=(n * 3, nl * 2.2),
                                 squeeze=False)
        for col, (sf, sp) in enumerate(zip(sel, snapshots)):
            it = int(sf.stem.split("_")[-1])
            for row, ln in enumerate(lnames):
                ax = axes[row, col]
                if ln not in sp:
                    # This snapshot doesn't have the layer — leave the cell
                    # blank and move on.  Otherwise we'd crash the report.
                    ax.set_axis_off()
                    continue
                S = sp[ln].numpy()
                S_n = S / S.max()
                ax.semilogy(S_n, linewidth=1.0, color="#2E86AB")
                ax.fill_between(range(len(S_n)), S_n,
                                alpha=0.1, color="#2E86AB")
                er = effective_rank(torch.tensor(S), self.cfg.erank_eps)
                if row == 0:
                    ax.set_title(f"{_fmt_total_steps(it * self._step_scale)}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(ln.split("_", 2)[-1][:20], fontsize=7)
                ax.text(0.95, 0.95, f"er={er:.1f}", transform=ax.transAxes,
                        fontsize=6, ha="right", va="top")
                ax.tick_params(labelsize=5)

        fig.suptitle("Weight Spectral Decay (actor)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"spectral_evolution.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_collapse_indicators(self, all_h, out):
        keys = [
            ("investigator/actor/preact_norm", "Pre-activation L2 norm"),
            ("investigator/actor/dead_neuron_frac", "Dead neuron fraction"),
            ("investigator/policy/action_mean_variance", "Policy variance"),
        ]
        fig, axes = plt.subplots(1, len(keys),
                                 figsize=(len(keys) * 4.5, 3.5))
        for ax, (k, t) in zip(axes, keys):
            all_its = []
            for label, h in all_h:
                if k in h:
                    its, vs = zip(*h[k])
                    ax.plot(its, vs, label=label, linewidth=1.2)
                    all_its.extend(its)
            ax.set_title(t, fontsize=9)
            _setup_sci_xaxis(ax, all_its)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Collapse Indicators", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"collapse_indicators.{self.cfg.plot_format}",
                    dpi=self.cfg.plot_dpi, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# Cross-experiment comparison utility
# ===================================================================

def compare_experiments(
    experiment_dirs: dict[str, str | Path],
    output_dir: str | Path = "comparison_plots",
):
    """Compare investigator outputs across experiments.

    Args:
        experiment_dirs: {label: path_to_investigator_dir}
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_h = []
    for label, d in experiment_dirs.items():
        hp = Path(d) / "history.json"
        if hp.exists():
            with open(hp) as f:
                all_h.append((label, json.load(f)))

    if not all_h:
        return

    # Feature erank comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    all_its = []
    for label, h in all_h:
        k = "investigator/actor/feature_erank"
        if k in h:
            its, vs = zip(*h[k])
            ax.plot(its, vs, label=label, linewidth=1.5)
            all_its.extend(its)
    _setup_sci_xaxis(ax, all_its)
    ax.set_ylabel("effective rank")
    ax.set_title("Actor Feature erank -- Cross-Architecture")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cmp_feature_erank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Gram side-by-side
    ne = len(experiment_dirs)
    fig, axes = plt.subplots(2, ne, figsize=(ne * 3.5, 7),
                             height_ratios=[3, 1], squeeze=False)
    for col, (label, d) in enumerate(experiment_dirs.items()):
        gdir = Path(d) / "grams"
        gfiles = sorted(gdir.glob("gram_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
        if not gfiles:
            continue
        gram = torch.load(gfiles[-1], weights_only=True)
        g_np = gram.numpy()
        it = int(gfiles[-1].stem.split("_")[1])

        _, Sg, _ = torch.linalg.svd(gram)
        er = effective_rank(Sg)

        ax_g = axes[0, col]
        norm_v, cmap = _gram_colornorm(g_np)
        ax_g.imshow(g_np, cmap=cmap, norm=norm_v,
                    aspect="equal", interpolation="nearest")
        ax_g.set_title(f"{label}\ner={er:.1f}", fontsize=9)

        ax_s = axes[1, col]
        Sn = (Sg / Sg.sum()).numpy()
        ax_s.semilogy(Sn, color="#2E86AB", linewidth=1.0)
        ax_s.axhline(1 / len(Sn), color="#A23B72",
                     linestyle="--", linewidth=.6)
        if col == 0:
            ax_s.set_ylabel("norm sigma", fontsize=7)

    fig.suptitle("Gram Comparison (final checkpoint)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "cmp_grams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison -> {out}")


# ===================================================================
# CLI for post-hoc
# ===================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--compare", nargs="*", default=None)
    parser.add_argument("--cross", nargs="*", default=None,
                        help="label:path pairs for cross-experiment")
    args = parser.parse_args()

    if args.cross:
        dirs = {}
        for item in args.cross:
            l, p = item.split(":", 1) if ":" in item \
                else (Path(item).parent.name, item)
            dirs[l] = p
        compare_experiments(dirs)
    else:
        inv = Investigator.__new__(Investigator)
        inv.cfg = InvestigatorCfg()
        inv._log_dir = Path(args.log_dir)
        inv._history = {}
        inv._sort_indices = None
        inv._step_scale = inv.cfg.num_steps_per_env * inv.cfg.num_envs
        obs_p = inv._log_dir / "fixed_eval_obs.pt"
        if obs_p.exists():
            inv._fixed_eval_obs = torch.load(obs_p, weights_only=True)
        inv.generate_report(compare_dirs=args.compare)
