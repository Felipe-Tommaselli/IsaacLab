# Mind The Phase (CORL 2026 Submission, under review)

[![Watch the demo](mind_the_phase_thumb.png)](mind_the_phase.mp4)

This fork is upstream [Isaac Lab](https://github.com/isaac-sim/IsaacLab) plus our research efforts on Mind The Phase. We use `rsl_rl` in all experiments.

---

## 1. What lives in this fork

### 1.1 Architectures

All architecture variants are RSL-RL (≥ 4.0) models/algorithms under [`source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/):

- **SimBa** (residual, normalized): [`utils/rsl_rl/models/simba.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/simba.py).
  In-house SimBa v1 adapted for PPO. See paper **Appendix A.1** for the exact block, init, and the depth-stable `α = 1/√N` branch scaling. Two integration modes: `SimbaModel` (subclasses RSL-RL `MLPModel`, swaps the trunk) and `SimbaActorCritic` (asymmetric actor/critic trunks).

- **MLP** (vanilla baseline): stock RSL-RL `MLPModel`.
  The `*-MLP-v0` gym IDs (modern `actor`/`critic`/`obs_groups` shape), **not** the legacy `*-v0`.

- **PFO** (Proximal Feature Optimization, Moalla et al.):[`utils/rsl_rl/algorithms/ppo_pfo.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/algorithms/ppo_pfo.py).
  Loss-term injection. Cfgs must set `self.algorithm = RslRlPpoWithPfoCfg(...)`, or `pfo_coef` is silently ignored (Hydra does not error on a missing field).

- **LinOp** (research experiment): [`utils/rsl_rl/models/linop.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/linop.py), with design notes in [`LINOP_INVESTIGATION.md`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/LINOP_INVESTIGATION.md).
  Controlled via `agent.actor.linop_num_factors`.

---

## 2. The Investigator

The Investigator quietly does the heavy lifting in Mind the Phase! Behind every rank figure in the paper (feature rank, Gram rank, weight rank, and the phase-conditioned policy-Jacobian rank of Figs. 3–8), all the data was processed by the Investigator. It is a **self-contained observability engine** that attaches to a live RSL-RL `OnPolicyRunner` **without modifying RSL-RL itself**, by monkey-patching at runtime.

We currently vendor `MLflow` and, especially, `W&B` (Wandb). These are the two backends the Investigator can log to; we use both to leave redundancy in the logging pipeline.

Source: [`scripts/reinforcement_learning/rsl_rl/investigator.py`](scripts/reinforcement_learning/rsl_rl/investigator.py) (VERY large — please prefer the docstrings rather than the whole file).

In [`train.py`](scripts/reinforcement_learning/rsl_rl/train.py) the entire integration is:

```python
investigator = Investigator(runner, cfg=InvestigatorCfg())
investigator.install(sort_fn=lambda obs: obs[:, 9:12].norm(dim=-1))
...
runner.learn(...)                # runs as normal, now instrumented
investigator.generate_report()   # post-hoc matplotlib report (supplement to W&B)
```

`obs[:, 9:12]` is the commanded planar velocity `(vx, vy, ωz)` in the observation layout (base lin-vel `0:3`, ang-vel `3:6`, projected gravity `6:9`, command `9:12`; see paper **Appendix C.2 → Observation Space**). Its norm is used as a sort key so the fixed eval batch — and therefore the Gram/Jacobian matrices — is ordered by command magnitude, which is what makes the block structure in the Gram visualizations (Fig. 8) legible.

### 2.1 How the patch works

`Investigator.install()` (docstring at [`investigator.py:574`](scripts/reinforcement_learning/rsl_rl/investigator.py#L574)) does three things:

1. Captures a fixed eval observation batch (`n_eval_obs`, default 2048) once, so every checkpoint's rank is measured on the same inputs and ranks are comparable across training time and across runs.
2. Wraps `runner.learn()`. The wrapper detects the RSL-RL version and patches the right logging hook:
   - RSL-RL **≥ 5.0**: patches `runner.logger.log(it, **kwargs)`.
   - RSL-RL **< 5.0**: patches `runner.log(locs, ...)` directly.

   In both cases it normalizes the call into a common `locs` dict, calls `investigator._on_iteration(it, locs)` to compute metrics, then delegates to the original logger and restores it in a `finally` (so a crash in analysis can never take down a training trial). This kind of redundancy shows up all around the code: since many runs are fired in parallel on our GPU cluster, several techniques were used to ensure robust telemetry and observability.
3. **Hooks** the penultimate layer (`register_forward_hook`) to capture the feature matrix `Fθ(X)` for feature/Gram rank, and computes the policy Jacobian via `torch.func.jacrev + vmap`.

> Note: the Investigator is pure observation and can be disabled with one cfg flag (`InvestigatorCfg.enabled = False`).

### 2.2 What it measures (and the cfg knobs)

All cadence/metric toggles live on `InvestigatorCfg` ([`investigator.py:157`](scripts/reinforcement_learning/rsl_rl/investigator.py#L157)):

- **Feature rank & PCA-99 rank**: penultimate activations, entropy effective rank (paper Eq. 3).
- **Gram rank**: cosine-similarity Gram of L2-normalized features, rows sorted by `sort_fn`.
- **Weight rank**: per-layer effective rank of the actor's linear weights (`weight_rank_interval`).
- **Policy-Jacobian effective rank**: `∂π/∂x` over a `jacobian_batch_size` subset; the central object of the paper.
- **Phase-conditioned rank**: pass `phase_fn=lambda env: ...` to `install()` to label each env by gait phase (e.g. thresholded foot contact forces) and split swing/stance/double-support. This is what exposes the swing-dominant Δφ split (Fig. 3) that the global average washes out. Gated by `phase_log_enabled` / `min_samples_per_phase`; silently skipped if no `phase_fn` is supplied (as in the default `train.py` install above, which passes only `sort_fn`).

More redundancy? Yes. Cheap live metrics run every `log_interval` (default 100 iters); expensive Gram/Jacobian/plots run every `checkpoint_interval` (default 1000). Metrics go to **W&B and/or MLflow** (`backend = "wandb" | "mlflow" | "multi"`), with local disk under `logs/.../investigator/` as backup, and `generate_report()` produces a post-hoc matplotlib supplement.

---

## 3. Infrastructure

The experiments are produced by two repositories that are developed together but kept separate on purpose:

| Repo | Role | Entry doc |
|---|---|---|
| **IsaacLab** (this repo, `isaacray-integration` branch) | Architectures (SimBa / LinOp / PFO / MLP), per-`(robot, terrain, architecture)` gym IDs + RunnerCfgs, the RSL-RL train script, and the **Investigator** rank analyzer. | this file |
| **IsaacRay** (orchestration harness, **not yet open-sourced**) | Parallel hyperparameter sweeps: a Dockerized Ray Tune + MLflow + W&B stack that mounts this checkout and fans trials across a heterogeneous GPU cluster. | [`ISAACRAY.md`](ISAACRAY.md) |

IsaacRay **does not vendor** IsaacLab. It bind-mounts this checkout into its training container and overlays its own `ray/` scripts; the only contract between them is that `ISAACLAB_PATH` points at a clone with `isaacray-integration` checked out. Without this branch the train script will not attach to MLflow/W&B runs and the Investigator is never installed. You do not need IsaacRay to reproduce a run, the train script is standalone; see [`ISAACRAY.md`](ISAACRAY.md) Section 4.

```
sweep.py launch-task spot_flat            (IsaacRay, host)
   └─ Ray Tune driver (IsaacRay/ray/tuner.py, in container)
        └─ one worker per GPU → subprocess:
             scripts/reinforcement_learning/rsl_rl/train.py   (THIS repo)
                ├─ builds the RSL-RL OnPolicyRunner over a velocity task
                ├─ Investigator.install(...)  ← monkey-patches runner.learn()
                └─ runner.learn()  → RSL-RL + rank metrics → MLflow / W&B
```

### 3.1 IsaacRay TL;DR

IsaacRay turns "run this architecture × this terrain × N seeds" into one command. It is a three-container Docker stack (training + MLflow + TensorBoard) that schedules trials with Ray Tune across all visible GPUs and scales to a heterogeneous multi-node cluster via Ray custom-resource tags. The five paper sweep targets: `anymal_flat`, `anymal_rough`, `h1_flat`, `h1_rough`, `spot_flat`, each expand the same **10-variant grid** (3 MLP + 3 SimBa + 1 PFO+MLP + 1 PFO+SimBa + 2 LinOp) over 5 seeds = 50 trials/sweep. It is **not yet open-sourced**; how it drives this repo, and how to reproduce the same results **standalone without it**, is in [`ISAACRAY.md`](ISAACRAY.md).

---

## 4. Reproducing the paper

You do not need IsaacRay, every trial is one `train.py` invocation. A single run:

```bash
# Spot flat, SimBa (2 blocks), seed 42 — one of the 50 spot_flat trials (Figs. 3, 5, 6, 7).
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Spot-Simba-v0 \
    --headless --num_envs 4096 --name spot_flat_simba2_s42 \
    agent.seed=42 agent.actor.simba_num_blocks=2 agent.max_iterations=2000

# Vanilla MLP baseline for the architecture contrast:
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Spot-MLP-v0 \
    --headless --num_envs 4096 --name spot_flat_mlp_s42 \
    agent.seed=42 agent.max_iterations=2000
```

The full sweep is this command looped over the 10 architecture gym IDs × 5 seeds per `(robot, terrain)`; the gym-ID list and per-terrain iteration caps are in [`ISAACRAY.md`](ISAACRAY.md) Section 2 / [`CHANGES_FROM_UPSTREAM.md`](CHANGES_FROM_UPSTREAM.md) Section 3. The over-training/collapse run (Fig. 7) is the same command with `agent.algorithm.num_learning_epochs=10`; the LR study (Fig. 9, App. B.2) sweeps `agent.algorithm.learning_rate`.

Rank/health metrics for every run are emitted by the Investigator to MLflow and/or W&B, with a local backup under `logs/rsl_rl/{name}/investigator/`. PPO/reward curves come from RSL-RL through the same channels.

An anonymized snapshot of an earlier version is referenced in the paper (**Appendix B.1**): <https://anonymous.4open.science/r/IsaacLab-F6E2>.

---

## 5. Citation

Mind the Phase is under review; cite the paper once a stable reference is available. Per upstream's request, also cite Isaac Lab itself ([`README.md` → Citation](README.md)) and SimBa (Lee et al., ICLR 2025).
