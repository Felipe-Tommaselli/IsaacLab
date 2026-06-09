# Mind The Phase (CORL 2026 Submission, under review)

[![Watch the demo](mind_the_phase_thumb.png)](mind_the_phase.mp4)

A research fork of [Isaac Lab](https://github.com/isaac-sim/IsaacLab). We left the simulator alone and built our Mind The Phase work on top of it, training every policy with `rsl_rl`.

## 1. What lives in this fork

### 1.1 Architectures

All the variants are RSL-RL (≥ 4.0) models living under [`source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/). You pick one from a task's gym ID, the rest will follow.

- **SimBa** (residual, normalized): [`utils/rsl_rl/models/simba.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/simba.py).
  Our in-house SimBa v1, adapted for PPO. The paper (**Appendix A.1**) has the exact block, init, and the depth-stable `α = 1/√N` branch scaling. It comes in two flavors: `SimbaModel` swaps the trunk of RSL-RL's `MLPModel`, and `SimbaActorCritic` gives the actor and critic separate trunks.

- **MLP** (vanilla baseline): stock RSL-RL `MLPModel`.
  Use the `*-MLP-v0` gym IDs, **not** the legacy `*-v0`.

- **PFO** (Proximal Feature Optimization, Moalla et al.): [`utils/rsl_rl/algorithms/ppo_pfo.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/algorithms/ppo_pfo.py).
  A loss term we fold into PPO to keep the learned features healthy. Watch out: cfgs have to set `self.algorithm = RslRlPpoWithPfoCfg(...)`, or `pfo_coef` is silently ignored, since Hydra won't complain about the missing field. This is just a research feature, not a mandatory component. 

- **LinOp** (research experiment): [`utils/rsl_rl/models/linop.py`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/linop.py), with design notes in [`LINOP_INVESTIGATION.md`](source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/LINOP_INVESTIGATION.md).
  An exploratory linear-operator variant we're still characterizing, driven by `agent.actor.linop_num_factors`.

## 2. The Investigator

The Investigator quietly does the heavy lifting in Mind the Phase. Every rank figure in the paper (feature rank, Gram rank, weight rank, and the phase-conditioned policy-Jacobian rank of Figs. 3–8) was measured by it. It is a self-contained observability engine that attaches to a live RSL-RL `OnPolicyRunner` **without touching RSL-RL itself**, by monkey-patching at runtime.

We log to both `MLflow` and `W&B` (Wandb). With many paralallel jobs in flight this redundancy has saved us more than once. 

Source: [`scripts/reinforcement_learning/rsl_rl/investigator.py`](scripts/reinforcement_learning/rsl_rl/investigator.py) (it's a big file, so lean on the docstrings rather than reading it top to bottom).

The integration in [`train.py`](scripts/reinforcement_learning/rsl_rl/train.py):

```python
investigator = Investigator(runner, cfg=InvestigatorCfg())
investigator.install(sort_fn=lambda obs: obs[:, 9:12].norm(dim=-1))
...
runner.learn(...)                # runs as normal, now instrumented
investigator.generate_report()   # post-hoc matplotlib report (supplement to W&B)
```

`obs[:, 9:12]` is the commanded planar velocity `(vx, vy, ωz)` (the observation layout is base lin-vel `0:3`, ang-vel `3:6`, projected gravity `6:9`, command `9:12`; see paper **Appendix C.2**). We sort the fixed eval batch by its norm, so the Gram and Jacobian matrices come out ordered by command magnitude. That ordering is what makes the block structure in the Gram plots (Fig. 8) actually readable.

### 2.1 How the patch works

`Investigator.install()` (docstring at [`investigator.py:574`](scripts/reinforcement_learning/rsl_rl/investigator.py#L574)) does three things:

1. Grabs a fixed eval observation batch (`n_eval_obs`, default 2048) once, so every checkpoint is scored on the same inputs and ranks stay comparable across training time and across runs.
2. Wraps `runner.learn()`. The wrapper checks the RSL-RL version and patches whichever logging hook is there:
   - RSL-RL **≥ 5.0**: patches `runner.logger.log(it, **kwargs)`.
   - RSL-RL **< 5.0**: patches `runner.log(locs, ...)` directly.

   either way it normalizes the call into a common `locs` dict, computes its metrics, then hands back to the original logger and restores it in a `finally`, so a crash in the analysis can never take down a training trial. That caution runs through the whole codebase.
3. **Hooks** the penultimate layer (`register_forward_hook`) to capture the feature matrix `Fθ(X)` for feature/Gram rank, and computes the policy Jacobian via `torch.func.jacrev + vmap`.

> Note: the Investigator is pure observation. It never touches the policy, and one cfg flag (`InvestigatorCfg.enabled = False`) turns it off.

### 2.2 What it measures

The cadence and metric toggles all live on `InvestigatorCfg` ([`investigator.py:157`](scripts/reinforcement_learning/rsl_rl/investigator.py#L157)):

- **Feature rank & PCA-99 rank**: how many directions the penultimate activations actually use, as entropy effective rank (paper Eq. 3).
- **Gram rank**: same idea through a cosine-similarity Gram of L2-normalized features, rows sorted by `sort_fn`.
- **Weight rank**: per-layer effective rank of the actor's linear weights (`weight_rank_interval`).
- **Policy-Jacobian effective rank**: `∂π/∂x` over a `jacobian_batch_size` subset. This is the central object of the paper.
- **Phase-conditioned rank**: pass `phase_fn=lambda env: ...` to `install()` to tag each env by gait phase (e.g. thresholded foot contacts) and split swing/stance/double-support. This is what surfaces the swing-dominant Δφ split (Fig. 3) that the global average hides. Gated by `phase_log_enabled` / `min_samples_per_phase`, and quietly skipped when no `phase_fn` is given (as in the default `train.py` above, which passes only `sort_fn`).

The two cadences keep the cost down (which we plan to measure properly in future iterations, but, empirically, it's lower than expected). the cheap live metrics run every `log_interval` (default 100 iters), the expensive Gram/Jacobian/plots only every `checkpoint_interval` (default 1000). Everything goes to **W&B and/or MLflow** (`backend = "wandb" | "mlflow" | "multi"`), with a disk backup under `logs/.../investigator/`, and `generate_report()` renders the offline matplotlib supplement.

## 3. Infrastructure

The experiments come out of two repos we develop together but keep separate on purpose:

| Repo | Role | Entry doc |
|---|---|---|
| **IsaacLab** (this repo, `isaacray-integration` branch) | The architectures (SimBa / LinOp / PFO / MLP), the per-`(robot, terrain, architecture)` gym IDs and RunnerCfgs, the RSL-RL train script, and the **Investigator**. | this file |
| **IsaacRay** (orchestration harness, **not yet open-sourced**) | The sweeps: a Dockerized Ray Tune + MLflow + W&B stack that mounts this checkout and fans trials across a heterogeneous GPU cluster. | [`ISAACRAY.md`](ISAACRAY.md) |

IsaacRay **doesn't vendor** IsaacLab. It bind-mounts this checkout into its training container and overlays its own `ray/` scripts; the only contract between them is that `ISAACLAB_PATH` points at a clone with `isaacray-integration` checked out. Without this branch the train script won't attach to MLflow/W&B and the Investigator never installs. But you don't need IsaacRay to reproduce a run, the train script stands on its own; see [`ISAACRAY.md`](ISAACRAY.md) Section 4.

```
sweep.py launch-task spot_flat            (IsaacRay, host)
   └─ Ray Tune driver (IsaacRay/ray/tuner.py, in container)
        └─ one worker per GPU → subprocess:
             scripts/reinforcement_learning/rsl_rl/train.py   (THIS repo)
                ├─ builds the RSL-RL OnPolicyRunner over a velocity task
                ├─ Investigator.install(...)  ← monkey-patches runner.learn()
                └─ runner.learn()  → RSL-RL + rank metrics → MLflow / W&B
```

### 3.1 TL;DR

IsaacRay turns "run this architecture × this terrain × N seeds" into one command. It's a three-container Docker stack (training + MLflow + TensorBoard) that schedules trials with Ray Tune across every visible GPU and scales to a multi-node cluster via Ray custom-resource tags. The five paper sweep targets, `anymal_flat`, `anymal_rough`, `h1_flat`, `h1_rough`, and `spot_flat`, each expand the same **10-variant grid** (3 MLP + 3 SimBa + 1 PFO+MLP + 1 PFO+SimBa + 2 LinOp) over 5 seeds, so 50 trials a sweep. It's not yet open-sourced; how it drives this repo, and how to reproduce the same results **without it**, is in [`ISAACRAY.md`](ISAACRAY.md).

## 4. Reproducing the paper

You don't need IsaacRay, every trial is just one `train.py` call. A single run:

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

The full sweep is that command looped over the 10 gym IDs × 5 seeds per `(robot, terrain)`; the gym-ID list and per-terrain iteration caps are in [`ISAACRAY.md`](ISAACRAY.md) Section 2 and [`CHANGES_FROM_UPSTREAM.md`](CHANGES_FROM_UPSTREAM.md) Section 3. The over-training/collapse run (Fig. 7) is the same command with `agent.algorithm.num_learning_epochs=10`; the LR study (Fig. 9, App. B.2) sweeps `agent.algorithm.learning_rate`.

Every run's rank and health metrics come from the Investigator into MLflow and/or W&B, with a local backup under `logs/rsl_rl/{name}/investigator/`. The PPO and reward curves come from RSL-RL through the same channels.

An anonymized snapshot is referenced in the paper (**Appendix B.1**): <https://anonymous.4open.science/r/IsaacLab-F6E2>.

## 5. Citation

Mind the Phase is under review; cite the paper once a stable reference is available. Per upstream's request, also cite Isaac Lab itself ([`README.md`](README.md)) and SimBa (ICLR 2025).
