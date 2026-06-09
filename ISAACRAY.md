# IsaacRay 

> **Status:** IsaacRay (the orchestration layer) is **not open-sourced yet**. This document describes how it drives the code in *this* repo so that the paper's sweeps can be reproduced, and so a reviewer can understand the experimental pipeline end to end. Everything the paper actually *measures* lives in this IsaacLab fork (see [`MIND_THE_PHASE.md`](MIND_THE_PHASE.md); IsaacRay is the harness that launches it at scale.

You do **not** need IsaacRay to reproduce a single run, the train script in this repo runs standalone (Section 4). IsaacRay only adds parallelism, seed sweeping, and unified logging on top of it.

---

## 1. What IsaacRay is

IsaacRay is a thin, Dockerized **Ray Tune + MLflow + Weights & Biases** harness that runs many `train.py` invocations in parallel across a multi-GPU machine (and, optionally, a small heterogeneous cluster). It is deliberately separate from IsaacLab: it **bind-mounts this checkout** into its training container and overlays its own Ray scripts on top, so the simulation/architecture/measurement code stays here and only the *orchestration* lives there.

The contract between the two repos is exactly one thing: IsaacRay expects this repo on the **`isaacray-integration`** branch, because that branch is what makes `train.py` (a) attach to an MLflow/W&B run and (b) install the Investigator. On any other branch the sweeps would run but produce no rank telemetry.

```
                      IsaacRay (host)                         THIS repo (mounted in container)
   sweep.py launch-task <alias>
        │  resolves alias → cfg + defaults (registry)
        ▼
   Ray Tune driver  ── one worker per GPU ──►  python train.py --task <gym-id> <hydra overrides>
        │                                            ├─ build RSL-RL OnPolicyRunner
        │                                            ├─ Investigator.install(...)   (rank hooks)
        │                                            └─ runner.learn()
        ▼                                                   │
   MLflow + W&B  ◄──────────── metrics (RSL-RL + Investigator) ──────────────┘
```

A trial is just `train.py` with a `--task` and a set of `agent.*` Hydra overrides. IsaacRay's whole job is to enumerate those override combinations, fan them across GPUs/seeds, and collect the logs.

---

## 2. The experiment grid

Every paper sweep is a single **10-variant architecture grid** repeated over **5 seeds** (`repeat_run_count=5`) → **50 trials/sweep**:

| # | Variant | Gym ID family | Hydra knobs that define it |
|---|---|---|---|
| 3 | MLP (small/mid/large depth) | `…-MLP-v0` | `agent.actor.hidden_dims` |
| 3 | SimBa (1/2/3 blocks) | `…-Simba-v0` | `agent.actor.simba_num_blocks` |
| 1 | PFO + MLP | `…-PFO-v0` | `agent.algorithm.pfo_coef` (+ `RslRlPpoWithPfoCfg`) |
| 1 | PFO + SimBa | `…-Simba-PFO-v0` | both of the above |
| 2 | LinOp (2 factorizations) | `…-LinOp-v0` | `agent.actor.linop_num_factors` |

This grid is instantiated per `(robot, terrain)`:

| Sweep alias | Robot / terrain | Iteration cap | Paper figures |
|---|---|---|---|
| `spot_flat` | Spot, flat | 2000 (reduced from the 20000 standalone baseline) | Figs. 3, 5, 6, 7 — the sim-to-real platform |
| `anymal_flat` | ANYmal-D, flat | 1500 | cross-robot (Figs. 5, 6c) |
| `anymal_rough` | ANYmal-D, rough | 4000 | perceptive locomotion |
| `h1_flat` | Unitree H1, flat | 2000 | humanoid (Figs. 5, 6c) |
| `h1_rough` | Unitree H1, rough | 4000 | humanoid, perceptive |

---

## 3. How a sweep is specified (IsaacRay side)

A sweep config is a small Python class (a `JobCfg`) that fills two dicts:

- `runner_args` — CLI flags for `train.py` (`--task`, `--num_envs`, `--headless`, …).
- `hydra_args` — Hydra overrides for the env/agent config (`agent.actor.hidden_dims`, `agent.seed`, `agent.max_iterations`, …).

Values are either plain Python or Ray Tune search primitives (`tune.grid_search`, `tune.sample_from`, `tune.loguniform`, …). The architecture grid is a `tune.grid_search` over a primary `agent.run_name` key, with every other architecture knob derived from it via `tune.sample_from` so the 10 variants stay internally consistent. When any `grid_search` is present the harness uses Ray's exhaustive `BasicVariantGenerator`; with no grid (e.g. the learning-rate study, paper Fig. 9 / App. B.2) it switches to Optuna/TPE Bayesian search.

The seed is injected per trial from a base seed, so the five repeats of each grid point differ only in seed — exactly the "five independent seeds per configuration" the paper reports.

> This is described here only so the pipeline is legible; the `JobCfg` files themselves live in the (currently private) IsaacRay repo. Reproducing the science does **not** require them — Section 4 shows the equivalent standalone commands.

---

## 4. Reproducing **without** IsaacRay (standalone)

Any single trial is reproducible from this repo alone, using the stock RSL-RL workflow. A trial is one architecture × one seed:

```bash
# Spot flat, SimBa (2 blocks), seed 42 — one of the 50 spot_flat trials.
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Spot-Simba-v0 \
    --headless --num_envs 4096 \
    --name spot_flat_simba2_s42 \
    agent.seed=42 \
    agent.actor.simba_num_blocks=2 \
    agent.max_iterations=2000
```

```bash
# Vanilla MLP baseline, same task/seed for the architecture contrast (Figs. 3, 6).
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Spot-MLP-v0 \
    --headless --num_envs 4096 \
    --name spot_flat_mlp_s42 \
    agent.seed=42 agent.max_iterations=2000
```

```bash
# Over-training / collapse run (Fig. 7): double the PPO epochs to force trust-region collapse.
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Spot-Simba-v0 \
    --headless --num_envs 4096 --name spot_flat_simba2_collapse_s42 \
    agent.seed=42 agent.max_iterations=2000 \
    agent.algorithm.num_learning_epochs=10
```

Each run writes to `logs/rsl_rl/{name}/` and — because this is the `isaacray-integration` branch — emits Investigator rank metrics (feature/Gram/weight/Jacobian and the phase-conditioned split) to your configured backend, with a local backup under `logs/rsl_rl/{name}/investigator/`. To reproduce a full sweep manually, loop the command over the architecture/seed combinations in the table in Section 2.

The Investigator's behavior, knobs, and the meaning of the `sort_fn`/`phase_fn` hooks are documented in [`MIND_THE_PHASE.md`](MIND_THE_PHASE.md) Section 3.

---

## 5. What IsaacRay adds on top (and what it intentionally does not)

**Adds:** parallel trial scheduling across all visible GPUs; a one-command launcher with a named-alias registry; unified MLflow + W&B + TensorBoard logging with per-project partitioning; resume-only-the-failed-trials; and optional multi-node fan-out across a heterogeneous GPU cluster via Ray custom-resource tags (so big variants land on high-VRAM cards and small ones on small cards).

**Does not:** vendor or fork Isaac Lab (it mounts this checkout at runtime); change any simulation, architecture, or measurement code. Every result is therefore attributable to code in *this* repo, which is why the open-sourced IsaacLab fork is sufficient to audit the method.

---

## 6. When IsaacRay is released

When the harness is open-sourced it will ship its own README covering setup, the sweep catalog, the registry, monitoring, multi-node patterns, and resume. Until then, this document plus the standalone commands in Section 4 are the supported reproduction path. The anonymized snapshot referenced in the paper (**App. B.1**, <https://anonymous.4open.science/r/IsaacLab-F6E2>) corresponds to an earlier state of this fork.
