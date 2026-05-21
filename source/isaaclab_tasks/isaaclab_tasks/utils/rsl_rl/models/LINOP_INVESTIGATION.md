# LinOp k Investigation — handoff notes

**Last updated:** 2026-05-20 ~17:38 UTC
**Branch:** HEAD (uncommitted changes in `IsaacLab` and `IsaacRay`)
**Files modified this round:**
- `IsaacLab/.../utils/rsl_rl/models/linop.py` — orthogonal init in `_FactoredLinear`
- `IsaacLab/.../config/anymal_d/agents/rsl_rl_ppo_linop_cfg.py` — updated comment, lr=3e-4, max_grad_norm=0.5
- `IsaacRay/.../tasks/anymal/flat_architecture_cfg.py` — main LinOp comparison now tracks k=2 and k=4; k=4 uses lr=1.5e-4
- `IsaacRay/.../tasks/anymal/linop8_init_std_cfg.py` — self-contained init_std sweep now includes std1p0/std0p5/std0p3
- `IsaacRay/.../tasks/anymal/linop8_lr_scale_cfg.py` — k=8 init_std=1.0 LR ablation over {7.5e-5, 1.5e-4, 3e-4}

## 2026-05-20 17:38 UTC update

Strategy shift based on Hu et al. and coworker feedback:
- Main locomotion claim should track `k=2` and `k=4`, not `k=8`.  The paper's results suggest k=4 is the useful low-rank-inducing sweet spot; k=8 is better treated as a stress test.
- The `k=4` main arm uses inverse-k LR scaling relative to the k=2 control: `3e-4 * 2/4 = 1.5e-4`.
- The k=8 optimization story is now tested directly with an LR ablation at init_std=1.0: `{7.5e-5, 1.5e-4, 3e-4}`.

Changed:
```python
# flat_architecture_cfg.py
LINOP_VARIANTS = [
    ("mlp3_linop2", 2, [512, 512, 512, 512, 256], 3e-4),
    ("mlp3_linop4", 4, [512, 512, 512, 512, 256], 1.5e-4),
]

# linop8_lr_scale_cfg.py
lr_variants = [
    ("linop8_lr0p000075_std1p0", 7.5e-5),
    ("linop8_lr0p00015_std1p0", 1.5e-4),
    ("linop8_lr0p0003_std1p0", 3e-4),
]
```

Retired the stale one-arm `linop8_lr_scale` tuner before it started workers.  Launched:
```bash
python3 sweep.py launch \
  ray/hyperparameter_tuning/tasks/anymal/flat_architecture_cfg.py \
  --cfg_class AnymalDFlatLinOpSweepJobCfg \
  --experiment_name linop_k2_k4_main \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500

python3 sweep.py launch \
  ray/hyperparameter_tuning/tasks/anymal/linop8_lr_scale_cfg.py \
  --cfg_class LinOp8LrScaleSweepJobCfg \
  --experiment_name linop8_lr_ablation_std1p0 \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500
```

`linop_k2_k4_main` has started at least one `mlp3_linop2` worker and queued k=4 workers.  `linop8_lr_ablation_std1p0` is launched; it may queue behind active jobs.

## 2026-05-20 16:24 UTC update

The prior `linop8_init_std` sweep completed, but it was missing the std1p0 baseline in code. Final outcomes:

| Variant | Final escaped seeds | Final stuck seeds | Iter-300 pass? |
|---|---:|---:|---|
| linop8_std0p5 | 2/5 reached reward ≈14 | 3/5 stayed near reward −0.7 to −2.0, len <24 | no |
| linop8_std0p3 | 3/5 reached reward 2.6–10.3 | 2/5 stayed near reward −0.9 to −1.2, len <22 | no |

This means lowering exploration alone is not sufficient, though std0p3 looks better than std0p5 by final recovery count.

Feedback applied:
- Put `linop8_std1p0` back into `linop8_init_std_cfg.py`.
- Added `linop8_lr_scale_cfg.py`, a one-arm optimization test: k=8, init_std=1.0, lr=7.5e-5 (`3e-4 * 2/8`), max_grad_norm=0.5.

Launched:
```bash
python3 sweep.py launch \
  ray/hyperparameter_tuning/tasks/anymal/linop8_init_std_cfg.py \
  --cfg_class LinOp8InitStdSweepJobCfg \
  --experiment_name linop8_init_std_self_contained \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500

python3 sweep.py launch \
  ray/hyperparameter_tuning/tasks/anymal/linop8_lr_scale_cfg.py \
  --cfg_class LinOp8LrScaleSweepJobCfg \
  --experiment_name linop8_lr_scale \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500
```

At launch, the init_std sweep occupied all 4 GPU slots. The LR-scale sweep is alive but pending in Ray until a worker frees.

Early checkpoint from the first active init_std batch (all four active trials are now past iter 300):

| Variant | Iter | Reward | Action std | Episode len | Log |
|---|---:|---:|---:|---:|---|
| linop8_std0p3 | 310 | −1.91 | 0.31 | 14.32 | 8681214009.log |
| linop8_std0p5 | 365 | −1.25 | 0.50 | 14.06 | 9211392510.log |
| linop8_std1p0 | 315 | −2.63 | 1.00 | 25.27 | 8780332412.log |
| linop8_std1p0 | 385 | −1.86 | 0.99 | 20.60 | 1692171567.log |

So far: 0/4 active init_std trials meet the iter-300 pass criterion. Continue to final 1500 iters before deciding, because previous reduced-std seeds sometimes escaped late.

---

## TL;DR

LinOp with `k=2` works (reaches reward ~19 by iter 950).
LinOp with `k=8` remains unreliable — even after fixing the Gaussian-init condition-number catastrophe (`cond ≈ 3×10⁹ → 1` via orthogonal init), some seeds get stuck at high action_std, short episodes, and negative reward; one std1p0 seed numerically blew up.

The current main paper-consistent path is `k=4` with inverse-k LR scaling (`1.5e-4`) against the known-good `k=2` control.  Keep `k=8` as a stress test via the LR ablation, not as the main locomotion claim.

---

## What was already fixed (prior rounds, already in code)

### Fix 1 — `std_type="log"` on the Gaussian distribution
Without this, `std` was a direct (positive) parameter, and Adam updates could push it negative. The log-parameterization (`std = exp(log_std)`) is monotonically positive **so long as `log_std` is finite**.

### Fix 2 — Orthogonal initialization for `_FactoredLinear`
**Diagnosis:** Random-Gaussian init of `k` factors, scaled per-factor for Kaiming output variance, produced a composed `W_eff = W_k @ … @ W_1` whose condition number was approximately:
- k=2: ~1.4×10⁶
- k=4: ~1.8×10⁸
- k=8: ~3.2×10⁹  ← rank-1 in floating point

The composed weight is essentially rank-1 at init, so gradient descent can only improve one singular direction at a time → multi-hundred-iteration "balancing phase" where reward stays near zero.

**Fix (in `linop.py`):**
```python
gain = 2.0 ** (1.0 / (2 * num_factors))
self.factors = nn.ParameterList(
    [nn.Parameter(torch.empty(dim, dim)) for _ in range(num_factors)]
)
for w in self.factors:
    nn.init.orthogonal_(w, gain=gain)
```
This gives: every singular value of `W_eff` = `gain^k = √2` → output variance = 2 (Kaiming target), condition number = 1 for any k. Verified empirically: `investigator/weights/actor/layer_2_factor_5_erank = 511.96` for k=8 ortho init (full rank).

### Fix 3 — Lower LR + tighter grad norm
`learning_rate: 1e-3 → 3e-4` and `max_grad_norm: 1.0 → 0.5` in `rsl_rl_ppo_linop_cfg.py`. Made k=8 trainable (no longer NaN-crashing on iter 1-2).

---

## The remaining problem (this round)

### Empirical state — `linop_ortho_full` sweep (currently running, 1500 iters, 5 seeds each variant)

At ~iter 800-950:

| Trial | Variant | Iter | Reward | Action std | Episode len |
|---|---|---:|---:|---:|---:|
| 9632099950 | linop2 | 926 | **19.25** | 0.42 | 980 |
| 4909899999 | linop2 | 951 | **19.75** | 0.42 | 991 |
| 3770662539 | linop8 | 790 | **−2.09** | **0.96** | **24.7** |
| 2996531306 | linop8 | 785 | −0.73 | 0.80 | 25.0 |

linop2 is healthy and converging well.
linop8 seed 3770662539 is **catastrophically stuck**: action_std barely moved from 1.0 → 0.96 in 790 iters, episode length is *still* ~25.
linop8 seed 2996531306 is slowly recovering but at 600+ iters it's still in negative territory and episode length is still trapped at 25.

### Reward trajectory comparison (every 50 iters)

linop2 seed 9632099950:
```
iter 50:  −0.91   iter 250: +0.01  iter 350: +8.52   iter 450: +13.36
iter 100: −0.73   iter 300: +4.03  iter 400: +11.75  iter 950: +19.25
```

linop8 seed 3770662539 (stuck):
```
iter 50:  −4.02   iter 250: −2.49  iter 450: −2.30
iter 100: −2.92   iter 300: −2.42  iter 790: −2.09   ← effectively flat for 700 iters
iter 200: −2.64   iter 400: −2.30
```

### Diagnosis — the zero-advantage attractor

The detailed per-iteration stats at iter 400 reveal the mechanism:

**linop8 (stuck)**:
- Mean action std: **0.98**  (initialized at 1.0, basically unchanged)
- Mean entropy loss: **16.84** (= entropy of a 12-D Gaussian with std≈1)
- Mean episode length: **26 steps** (robot falls almost immediately every episode)
- `track_lin_vel_xy_exp`: **0.0039** (robot is not walking, at all)
- `Metrics/base_velocity/error_vel_xy`: 0.052 (low because robot is just standing/falling, not because tracking works)

**linop2 at the same point (iter 300)**:
- Mean action std: 0.64
- Mean entropy loss: 11.72
- Mean episode length: **718**
- `track_lin_vel_xy_exp`: 0.46 (~46% of target velocity)

The trap is a feedback loop:
1. Policy is high-entropy → random actions
2. Random actions → robot falls within ~26 steps
3. All episodes return reward ≈ −2.5 (uniform across the batch)
4. Advantage ≈ 0 across all states → surrogate loss ≈ 0
5. With surrogate ≈ 0, the entropy bonus (`entropy_coef × H(π) = 0.005 × 17 ≈ 0.085`) becomes the dominant term — actively rewards maintaining high entropy
6. The policy never reduces std → stays in the trap forever

### Why the trap is seed-dependent

Comparing the two linop8 seeds:
- Seed 3770662539: iter 1 reward = −0.44, dipped to **−6.34** at iter 4, settled into the trap.
- Seed 2996531306: iter 1 reward = −0.53, dipped to **−41.08** at iter 15, recovered.

The seed with the **catastrophic** initial dip recovered fast because the −41 reward created a clear, unambiguous gradient ("avoid these actions"). The seed with the **moderate** initial dip got stuck because reward ≈ −2.5 across all trajectories looks the same to PPO.

### Why k=8 destabilizes more than k=2 on the first PPO step

After one Adam step (which normalizes the per-coordinate gradient to ≈ sign(grad)), each factor `W_i` moves by approximately `lr` in some direction. The composed update is:
```
δW_eff ≈ Σⱼ (W_k…W_{j+1}) · δW_j · (W_{j-1}…W_1)
```

With ortho init each rotational pre/postmultiplication is an isometry, so the Frobenius-norm contribution of each term is `~lr · dim`. The sum over `k` factors is `k · lr · dim`:

| k | ‖δW_eff‖_F at lr=3e-4, dim=512 |
|---:|---:|
| 2 | 0.31 |
| 8 | **1.23** (4× larger) |

That 4× larger first-step swing in the hidden weights is what pushes some k=8 seeds past the recoverable threshold.

---

## Current experiments (running now)

### Main k=2 vs k=4 sweep
**File:** `IsaacRay/ray/hyperparameter_tuning/tasks/anymal/flat_architecture_cfg.py`

10 trials total (2 variants x 5 seeds):
```python
LINOP_VARIANTS = [
    ("mlp3_linop2", 2, [512, 512, 512, 512, 256], 3e-4),
    ("mlp3_linop4", 4, [512, 512, 512, 512, 256], 1.5e-4),
]
```

This is now the main paper-consistent locomotion comparison.

### Self-contained init_std sweep
**File:** `IsaacRay/ray/hyperparameter_tuning/tasks/anymal/linop8_init_std_cfg.py`

Now 15 trials total (3 variants × 5 seeds):
```python
INIT_STD_VARIANTS = [
    ("linop8_std1p0", 1.0),
    ("linop8_std0p5", 0.5),
    ("linop8_std0p3", 0.3),
]
```

This keeps the exploration-level comparison self-contained. The earlier 0.5/0.3-only run showed that reduced init_std can help some seeds escape, but neither value met the original pass criterion by iter 300 across all seeds.

### LR-scaled optimization arm
**File:** `IsaacRay/ray/hyperparameter_tuning/tasks/anymal/linop8_lr_scale_cfg.py`

15 trials total (3 variants x 5 seeds), all k=8 with init_std=1.0:
```python
lr_variants = [
    ("linop8_lr0p000075_std1p0", 7.5e-5),
    ("linop8_lr0p00015_std1p0", 1.5e-4),
    ("linop8_lr0p0003_std1p0", 3e-4),
]
```

This tests whether inverse-k LR scaling fixes the high-entropy attractor without lowering exploration.

**Pass criterion:** by iter 300, mean reward ≥ 0 across all linop8 seeds. The more useful discriminator is seed survival rate by iter 1500: does an arm reduce stuck seeds from the observed 40-60% range to 0/5?

---

## Code state cheat-sheet

### `linop.py` — `_FactoredLinear.__init__`
```python
gain = 2.0 ** (1.0 / (2 * num_factors))  # ortho init
self.factors = nn.ParameterList([nn.Parameter(torch.empty(dim, dim)) for _ in range(num_factors)])
for w in self.factors:
    nn.init.orthogonal_(w, gain=gain)
```

### `rsl_rl_ppo_linop_cfg.py` — `__post_init__`
```python
self.algorithm.learning_rate = 3e-4
self.algorithm.max_grad_norm = 0.5
# init_std is STILL 1.0 here — that is the variable the new sweep overrides via Hydra.
self.actor = RslRlLinOpModelCfg(
    ...,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    linop_num_factors=2,
)
```

### `flat_architecture_cfg.py` — LINOP_VARIANTS
```python
LINOP_VARIANTS = [
    ("mlp3_linop2", 2, [512, 512, 512, 512, 256], 3e-4),
    ("mlp3_linop4", 4, [512, 512, 512, 512, 256], 1.5e-4),
]
```

### `linop8_init_std_cfg.py` (new)
```python
INIT_STD_VARIANTS = [
    ("linop8_std1p0", 1.0),
    ("linop8_std0p5", 0.5),
    ("linop8_std0p3", 0.3),
]
# Sweep sets agent.actor.distribution_cfg.init_std via Hydra,
# fixes k=8, lr=3e-4, ortho init (from model code).
```

### `linop8_lr_scale_cfg.py` (new)
```python
lr_variants = [
    ("linop8_lr0p000075_std1p0", 7.5e-5),
    ("linop8_lr0p00015_std1p0", 1.5e-4),
    ("linop8_lr0p0003_std1p0", 3e-4),
]
```

---

## What to check next session

1. **Quick status check:** are the running sweeps still alive?
   ```
   docker exec isaacray-training bash -c 'ls -lt /tmp/isaacray_trials/*.log | head -15'
   ```
   Look for actively-growing log files (mtime within ~3 min).

2. **`linop_ortho_full` final results** (10 trials, 1500 iters): how many linop8 seeds got stuck (action_std ~ 1.0 at end) vs recovered? Expectation: 40-60% stuck, the rest reach +5 to +15.

3. **`linop8_init_std_self_contained` results** — for each of the 15 trials (3 variants × 5 seeds), check:
    - Did reward cross 0 by iter 300?
    - What's the final action_std?
    - Final reward at iter 1500 — does it match or beat linop2's ~19?

4. **`linop_k2_k4_main` results** — for each of the 10 main trials, check:
    - Does k=4 with lr=1.5e-4 match or beat k=2?
    - Does k=4 avoid the k=8 stuck-seed pattern?

5. **`linop8_lr_ablation_std1p0` results** — for each of the 15 lr-scaled trials, check:
    - Did reward cross 0 by iter 300 with init_std still at 1.0?
    - Does lr=7.5e-5 or 1.5e-4 avoid stuck seeds, or does it only slow down learning?

   Quick check command:
   ```
   docker exec isaacray-training bash -c '
   for f in /tmp/isaacray_trials/*.log; do
     v=$(grep -aE "Run name:.*(mlp3_linop|linop8_std|linop8_lr0p)" "$f" | head -1 | awk "{print \$NF}")
     [ -z "$v" ] && continue
     iters=$(grep -ac "Learning iteration" "$f")
     r=$(grep -a "Mean reward:" "$f" | tail -1 | awk "{print \$3}")
     std=$(grep -a "Mean action std:" "$f" | tail -1 | awk "{print \$NF}")
     printf "%-15s iters=%4d  reward=%6s  std=%4s  %s\n" "$v" "$iters" "$r" "$std" "$(basename $f)"
   done | sort'
   ```

6. **If one init_std arm works:** decision needed — should that `init_std` become the default for LinOp in the base config (`rsl_rl_ppo_linop_cfg.py`), or stay a per-variant Hydra override?

7. **If neither init_std nor LR scaling works** (stuck attractor persists): the next lever is `entropy_coef`. Current is 0.005 (inherited from base PPO cfg). Try 0.001 or 0.0005 specifically for LinOp — this directly addresses the "entropy bonus dominates when advantage ≈ 0" failure mode.

8. **Investigator metrics from wandb:** by the time the 1500-iter sweep finishes, `investigator/policy/jacobian_erank`, `investigator/gram/erank` etc. should all have curves (this was the *original* symptom that started the whole investigation — the metrics were showing literal `[[ ${x}: ${y} ]]` strings because trials crashed before `checkpoint_interval=500`). Worth verifying the original symptom is resolved.

---

## Files & paths reference

- Model code: `/home/ray/Repos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/rsl_rl/models/linop.py`
- Runner cfg: `/home/ray/Repos/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_linop_cfg.py`
- Architecture sweep: `/home/ray/Repos/IsaacRay/ray/hyperparameter_tuning/tasks/anymal/flat_architecture_cfg.py`
- LinOp sweep module: `/home/ray/Repos/IsaacRay/ray/hyperparameter_tuning/tasks/anymal/linop_cfg.py`
- Trial logs (in container): `/tmp/isaacray_trials/*.log`
- Sweep launcher: `python3 sweep.py launch-task ...` from `/home/ray/Repos/IsaacRay/`
- MLflow: `http://localhost:5000`

Current registered launch aliases:

```bash
python3 sweep.py launch-task anymal_linop_main --max_iterations 1500
python3 sweep.py launch-task anymal_linop8_init_std --max_iterations 1500
python3 sweep.py launch-task anymal_linop8_lr_ablation --max_iterations 1500
```

Historical note: earlier entries in this document mention direct config files
such as `linop8_init_std_cfg.py` and `linop8_lr_scale_cfg.py`. Those were
consolidated into `linop_cfg.py`; the old paths should be treated as historical
launch records only. They remain as tiny compatibility shims so already-staged
Ray wrappers from the active sweeps can keep importing their original config
paths until those jobs finish.

## Architecture comparison (other families, for reference)

From the prior full sweep, 5 seeds × 1500 iters, mean final reward:

| Variant | Reward |
|---|---:|
| simba3 | 24.39 |
| pfo_simba3 | 24.34 |
| simba2 | 24.16 |
| simba1 | 23.93 |
| mlp2 | 22.95 |
| mlp1 | 22.50 |
| pfo_mlp3 | 22.31 |
| mlp3 | 21.93 |
| **linop2** | **21.48** (matches mlp3 baseline ✓) |
| **linop8** | **11.75** Gaussian init / hopefully now matching linop2 with ortho+init_std fix |

The LinOp paper's whole thesis is that k>1 should be at least as good as k=1 (same function class, biased toward low-rank). Right now we can't verify that claim — k=8 is strictly worse. Once init_std fix lands, that should change.

---

## 2026-05-20 official-run ledger update

Snapshot time: `2026-05-20 18:13:47 UTC` (`2026-05-20 15:13:47 America/Sao_Paulo`).

Purpose of this update:

- Preserve the exact official 5-seed runs so they can be recovered without relaunching.
- Track the paper-facing pivot from "make k=8 work" to "evaluate the paper-consistent k=4 sweet spot, while keeping k=8 as a stress-test ablation."
- Record the optimization-level lr-scaling arm suggested by the Appendix J / first-step amplification argument.

### Current plan / to-do list

1. Let `linop_k2_k4_main` finish as the main paper-facing LinOp comparison.
   - This is now the official clean comparison: k=2 vs k=4, 5 seeds each, 1500 iterations.
   - k=4 is the paper-consistent candidate. It uses lr `1.5e-4`, inverse-k scaled relative to k=2's `3e-4`.
   - Keep k=2 because it already matched the MLP baseline and anchors the comparison.

2. Let `linop8_init_std_self_contained` finish, but treat it as the k=8 stress-test / exploration-level fix.
   - This sweep now includes `init_std=1.0` again, so the comparison is self-contained: `1.0`, `0.5`, `0.3`.
   - The current evidence is mixed: lower `init_std` helps some seeds but does not fully eliminate stuck seeds.

3. Keep `linop8_lr_ablation_std1p0` queued/running as the optimization-level test.
   - This directly tests the coworker suggestion: scale lr down for k=8 while keeping `init_std=1.0`.
   - If lr scaling alone fixes k=8, the cleaner paper story is "optimize the amplification, do not hide it with lower exploration."
   - If lr scaling does not fix k=8, the paper story should emphasize that locomotion appears to prefer the lower-rank k=4 regime.

4. After the active sweeps finish, append final per-seed rows here.
   - Required columns: experiment name, variant, k, seed, lr, init_std, log ID, final iteration, final reward, final action std, final episode length, status.
   - Also pull MLflow/W&B run IDs if available.

5. Do not relaunch any of these official 5-seed arms unless the corresponding sweep never emits a durable W&B run or a run crashes before producing usable data.
   - W&B run IDs/URLs are the durable source of truth.
   - `/tmp/isaacray_trials/*.log` names are useful live breadcrumbs only; they may disappear when the compose stack is rebooted.
   - `tuner_latest.log` can be overwritten by whichever tuner launched most recently, so it is useful only for the latest tuner metadata.

6. If resource contention blocks the k=8 lr ablation from starting, first wait for the active k2/k4 and init_std workers to drain.
   - Only stop older workers after recording their final rows here.
   - The current priority order is: k2/k4 main > k8 lr ablation > k8 init_std remainder.

### Sweep launch records

#### Main paper comparison: k=2 vs k=4

Launched from `/home/ray/Repos/IsaacRay`:

```bash
python3 sweep.py launch ray/hyperparameter_tuning/tasks/anymal/flat_architecture_cfg.py \
  --cfg_class AnymalDFlatLinOpSweepJobCfg \
  --experiment_name linop_k2_k4_main \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500
```

Recorded launcher metadata:

- Experiment name: `linop_k2_k4_main`
- Sweep wrapper: `/tmp/isaacray_sweeps/e4336836/wrapped_cfg.py`
- Sweep ID: `e4336836`
- Seed range: `[41020, 1041020)`
- Expected trials: `10` (`2 variants x 5 seeds`)
- Variants:
  - `mlp3_linop2`: k `2`, hidden dims `[512, 512, 512, 512, 256]`, lr `3e-4`
  - `mlp3_linop4`: k `4`, hidden dims `[512, 512, 512, 512, 256]`, lr `1.5e-4`

Active/observed trial handles at snapshot:

| Experiment | Variant | k | lr | Seed | Trial log | Latest iter | Latest reward | Latest action std | Latest episode len | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `3e-4` | `182236` | `2233736496.log` | 716 | 17.71 | 0.47 | 986.43 | active, strong |
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `3e-4` | `55898` | `3225767421.log` | 686 | 17.65 | 0.46 | 982.21 | active, strong |
| `linop_k2_k4_main` | `mlp3_linop4` | 4 | `1.5e-4` | `837442` | `1715168587.log` | 660 | 13.95 | 0.60 | 981.57 | active, stable |
| `linop_k2_k4_main` | `mlp3_linop4` | 4 | `1.5e-4` | `952264` | `7467068784.log` | 660 | 13.09 | 0.59 | 953.92 | active, stable |

Interpretation at snapshot:

- k=2 is behaving like the prior good LinOp baseline: reward already around `17.7` before iteration 750.
- k=4 is not stuck. It is behind k=2 early, but it has healthy episode lengths and rewards around `13-14` by iteration 660.
- This is the most important sweep for the paper-facing answer to "does LinOp help locomotion policies?"

#### k=8 exploration-level init_std sweep

Launched from `/home/ray/Repos/IsaacRay`:

```bash
python3 sweep.py launch ray/hyperparameter_tuning/tasks/anymal/linop8_init_std_cfg.py \
  --cfg_class LinOp8InitStdSweepJobCfg \
  --experiment_name linop8_init_std_self_contained \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500
```

Recorded launcher metadata:

- Experiment name: `linop8_init_std_self_contained`
- Sweep wrapper: `/tmp/isaacray_sweeps/ca51e9dc/wrapped_cfg.py`
- Sweep ID: `ca51e9dc`
- Seed range: `[14011, 1014011)`
- Expected trials: `15` (`3 init_std variants x 5 seeds`)
- Fixed architecture: k `8`, hidden dims `[512, 512, 512, 512, 256]`
- Fixed lr: `3e-4`
- Variants:
  - `linop8_std1p0`: `init_std=1.0`
  - `linop8_std0p5`: `init_std=0.5`
  - `linop8_std0p3`: `init_std=0.3`

Observed trial handles at snapshot:

| Experiment | Variant | k | lr | init_std | Seed | Trial log | Latest iter | Latest reward | Latest action std | Latest episode len | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `3e-4` | 1.0 | `631185` | `1692171567.log` | 1500 | -0.67 | 0.79 | 84.42 | complete, weak/stuck |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `3e-4` | 1.0 | `255935` | `8780332412.log` | 1500 | -1.41 | 0.87 | 31.34 | complete, stuck |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `3e-4` | 1.0 | `454112` | `2153085586.log` | 9 | catastrophic negative | 1.00 | 37.56 | failed/blow-up |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `3e-4` | 1.0 | `181372` | `5045786537.log` | 195 | -6.23 | 1.01 | 30.06 | active, poor |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `3e-4` | 1.0 | `105487` | pending | pending | pending | pending | pending | queued/not observed |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `3e-4` | 0.5 | `998581` | `9211392510.log` | 1500 | 15.34 | 0.44 | 942.58 | complete, strong |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `3e-4` | 0.5 | `82431` | `4128101964.log` | 1500 | 1.70 | 0.45 | 556.83 | complete, partial recovery |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `3e-4` | 0.5 | `425776` | `9249468088.log` | 1300 | -1.07 | 0.49 | 15.73 | active, stuck/slow |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `3e-4` | 0.5 | `395483` | pending | pending | pending | pending | pending | queued/not observed |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `3e-4` | 0.5 | `257646` | pending | pending | pending | pending | pending | queued/not observed |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `3e-4` | 0.3 | `652947` | `8681214009.log` | 1500 | -1.24 | 0.33 | 15.04 | complete, stuck |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `3e-4` | 0.3 | `299969` | `9369181262.log` | 1500 | 7.44 | 0.31 | 960.87 | complete, recovered |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `3e-4` | 0.3 | `290447` | `2544665969.log` | 1375 | -0.42 | 0.31 | 43.34 | active, weak/stuck |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `3e-4` | 0.3 | `680414` | `7012937996.log` | 205 | -2.75 | 0.30 | 22.17 | active, early/poor |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `3e-4` | 0.3 | `872230` | pending | pending | pending | pending | pending | queued/not observed |

Interpretation at snapshot:

- Restoring `std1p0` was important because the self-contained comparison now shows the baseline k=8 behavior remains poor under the current optimizer settings.
- `std0p5` has the best single seed so far (`15.34`), but it also has weak/stuck seeds.
- `std0p3` has one recovered seed (`7.44`) and multiple weak/stuck seeds.
- Lowering exploration helps some k=8 seeds but does not yet look like a robust fix.

#### k=8 optimization-level lr ablation

Launched from `/home/ray/Repos/IsaacRay`:

```bash
python3 sweep.py launch ray/hyperparameter_tuning/tasks/anymal/linop8_lr_scale_cfg.py \
  --cfg_class LinOp8LrScaleSweepJobCfg \
  --experiment_name linop8_lr_ablation_std1p0 \
  --num_samples -1 --repeat_run_count 5 \
  --max_iterations 1500
```

Recorded launcher metadata from `/tmp/isaacray_last_sweep.json`:

```json
{
  "experiment_name": "linop8_lr_ablation_std1p0",
  "mlflow_uri": "http://localhost:5000",
  "num_samples": -1,
  "repeat_run_count": 5,
  "wrapper_path": "/tmp/isaacray_sweeps/6baa2c8a/wrapped_cfg.py",
  "wrapper_class": "linop8_lr_ablation_std1p0_LinOp8LrScaleSweepJobCfg",
  "total_expected": 15,
  "launched_at": "2026-05-20 14:37"
}
```

Additional launch record:

- Experiment name: `linop8_lr_ablation_std1p0`
- Sweep wrapper: `/tmp/isaacray_sweeps/6baa2c8a/wrapped_cfg.py`
- Sweep ID: `6baa2c8a`
- Seed range: `[64421, 1064421)`
- Expected trials: `15` (`3 lr variants x 5 seeds`)
- Fixed architecture: k `8`, hidden dims `[512, 512, 512, 512, 256]`
- Fixed init_std: `1.0`
- Fixed max_grad_norm: `0.5`
- Variants:
  - `linop8_lr0p000075_std1p0`: lr `7.5e-5`
  - `linop8_lr0p00015_std1p0`: lr `1.5e-4`
  - `linop8_lr0p0003_std1p0`: lr `3e-4`

Observed trial handles at snapshot:

| Experiment | Variant | k | lr | init_std | Seed | Trial log | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `linop8_lr_ablation_std1p0` | `linop8_lr0p000075_std1p0` | 8 | `7.5e-5` | 1.0 | pending | pending | launched, no worker log observed yet |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p00015_std1p0` | 8 | `1.5e-4` | 1.0 | pending | pending | launched, no worker log observed yet |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p0003_std1p0` | 8 | `3e-4` | 1.0 | pending | pending | launched, no worker log observed yet |

Interpretation at snapshot:

- This sweep is launched but likely waiting on resources from the active k2/k4 and k8 init_std workers.
- No `/tmp/isaacray_trials/*.log` with run names matching `linop8_lr0p...` has appeared yet.
- Keep checking the trial log directory before deciding whether to relaunch. The latest `tuner_latest.log` belongs to this sweep, but worker logs are the reliable trial handles.

### Latest log directory snapshot

Command used:

```bash
docker exec isaacray-training bash -c 'ls -lt /tmp/isaacray_trials/*.log | head -15'
```

Result at snapshot:

```text
-rw-r--r-- 1 root root  894209 May 20 18:13 /tmp/isaacray_trials/1715168587.log
-rw-r--r-- 1 root root  959175 May 20 18:13 /tmp/isaacray_trials/2233736496.log
-rw-r--r-- 1 root root  318945 May 20 18:13 /tmp/isaacray_trials/5045786537.log
-rw-r--r-- 1 root root  330157 May 20 18:13 /tmp/isaacray_trials/7012937996.log
-rw-r--r-- 1 root root  888289 May 20 18:13 /tmp/isaacray_trials/7467068784.log
-rw-r--r-- 1 root root 1732213 May 20 18:13 /tmp/isaacray_trials/2544665969.log
-rw-r--r-- 1 root root  923835 May 20 18:13 /tmp/isaacray_trials/3225767421.log
-rw-r--r-- 1 root root 1620240 May 20 18:13 /tmp/isaacray_trials/9249468088.log
-rw-r--r-- 1 root root 1790946 May 20 17:53 /tmp/isaacray_trials/4128101964.log
-rw-r--r-- 1 root root 1790780 May 20 17:52 /tmp/isaacray_trials/9369181262.log
-rw-r--r-- 1 root root 1790050 May 20 17:11 /tmp/isaacray_trials/8681214009.log
-rw-r--r-- 1 root root   34475 May 20 17:11 /tmp/isaacray_trials/2153085586.log
-rw-r--r-- 1 root root 1791881 May 20 17:08 /tmp/isaacray_trials/8780332412.log
-rw-r--r-- 1 root root 1792231 May 20 17:05 /tmp/isaacray_trials/1692171567.log
-rw-r--r-- 1 root root 1791157 May 20 17:04 /tmp/isaacray_trials/9211392510.log
```

### Durable W&B run ledger

This table is the recovery handle to use after `/tmp/isaacray_trials/*.log` is gone. The `/tmp` log ID is retained only as a temporary breadcrumb for matching the current live process.

W&B project for all observed runs below: `tommaselli/anymal_flat_v6`.

#### Main k=2/k=4 official runs

| Experiment | Variant | k | Seed | W&B run ID | W&B URL | W&B start stamp | Temporary log |
|---|---:|---:|---:|---:|---|---:|---:|
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `182236` | `uk3ogrvq` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/uk3ogrvq | `20260520_174007` | `2233736496.log` |
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `55898` | `22b516km` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/22b516km | `20260520_174158` | `3225767421.log` |
| `linop_k2_k4_main` | `mlp3_linop4` | 4 | `837442` | `op06nvh0` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/op06nvh0 | `20260520_174146` | `1715168587.log` |
| `linop_k2_k4_main` | `mlp3_linop4` | 4 | `952264` | `5cwkp6o9` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/5cwkp6o9 | `20260520_174203` | `7467068784.log` |

Local W&B paths observed inside the container:

```text
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-38-11_mlp3_linop2/wandb/run-20260520_174007-uk3ogrvq
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-39-09_mlp3_linop2/wandb/run-20260520_174158-22b516km
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-38-45_mlp3_linop4/wandb/run-20260520_174146-op06nvh0
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-39-42_mlp3_linop4/wandb/run-20260520_174203-5cwkp6o9
```

#### k=8 init_std official/stress-test runs

| Experiment | Variant | k | Seed | W&B run ID | W&B URL | W&B start stamp | Temporary log |
|---|---:|---:|---:|---:|---|---:|---:|
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `631185` | `ysfuaw7c` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/ysfuaw7c | `20260520_162258` | `1692171567.log` |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `255935` | `v6ort5e1` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/v6ort5e1 | `20260520_162418` | `8780332412.log` |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `454112` | `ql3yxj84` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/ql3yxj84 | `20260520_171100` | `2153085586.log` |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `181372` | `9dh8vkn2` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/9dh8vkn2 | `20260520_175637` | `5045786537.log` |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `105487` | pending | pending | pending | pending |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `998581` | `itotj6tf` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/itotj6tf | `20260520_162324` | `9211392510.log` |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `82431` | `0e2gvocu` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/0e2gvocu | `20260520_170527` | `4128101964.log` |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `425776` | `yu25o6v6` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/yu25o6v6 | `20260520_171409` | `9249468088.log` |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `395483` | pending | pending | pending | pending |
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `257646` | pending | pending | pending | pending |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `652947` | `own1vygg` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/own1vygg | `20260520_162353` | `8681214009.log` |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `299969` | `aensm442` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/aensm442 | `20260520_170657` | `9369181262.log` |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `290447` | `0m210exq` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/0m210exq | `20260520_171401` | `2544665969.log` |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `680414` | `9d8os113` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/9d8os113 | `20260520_175538` | `7012937996.log` |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `872230` | pending | pending | pending | pending |

Local W&B paths observed inside the container:

```text
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_16-22-44_linop8_std1p0/wandb/run-20260520_162258-ysfuaw7c
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_16-23-59_linop8_std1p0/wandb/run-20260520_162418-v6ort5e1
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-10-41_linop8_std1p0/wandb/run-20260520_171100-ql3yxj84
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-54-37_linop8_std1p0/wandb/run-20260520_175637-9dh8vkn2
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_16-23-09_linop8_std0p5/wandb/run-20260520_162324-itotj6tf
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-05-01_linop8_std0p5/wandb/run-20260520_170527-0e2gvocu
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-13-49_linop8_std0p5/wandb/run-20260520_171409-yu25o6v6
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_16-23-34_linop8_std0p3/wandb/run-20260520_162353-own1vygg
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-06-33_linop8_std0p3/wandb/run-20260520_170657-aensm442
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-13-36_linop8_std0p3/wandb/run-20260520_171401-0m210exq
/workspace/isaaclab/logs/rsl_rl/anymal_d_flat_linop/2026-05-20_17-53-40_linop8_std0p3/wandb/run-20260520_175538-9d8os113
```

#### k=8 lr ablation durable IDs

No W&B run IDs observed yet for `linop8_lr_ablation_std1p0` at the snapshot. The sweep is launched, but no worker log matching `linop8_lr0p...` has appeared yet.

Expected durable rows to fill once workers start:

| Experiment | Variant | k | lr | init_std | Seed | W&B run ID | W&B URL | W&B start stamp |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `linop8_lr_ablation_std1p0` | `linop8_lr0p000075_std1p0` | 8 | `7.5e-5` | 1.0 | pending | pending | pending | pending |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p00015_std1p0` | 8 | `1.5e-4` | 1.0 | pending | pending | pending | pending |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p0003_std1p0` | 8 | `3e-4` | 1.0 | pending | pending | pending | pending |

### Paper-facing decision framing

There are now two complementary questions:

1. **Main paper question:** does LinOp help locomotion policies in the low-rank regime predicted by the paper?
   - The right evidence is the k=2/k=4 official 5-seed sweep.
   - k=4 is the Hu et al.-consistent target: large enough to test rank induction, small enough to avoid the observed k=8 optimization pathologies.

2. **Stress-test question:** can k=8 be made to work, and if yes, why?
   - `init_std` sweep tests an exploration-level workaround.
   - `lr` sweep tests an optimization-level cause: first-step amplification grows with k, so lr should shrink roughly as `1/k`.
   - If lr scaling works at `init_std=1.0`, that is the cleaner mechanistic result.
   - If lr scaling does not work, k=8 should be described as an over-parameterized stress regime for this locomotion task rather than the recommended operating point.

---

## 2026-05-20 18:46 UTC live status / next actions

Snapshot time: `2026-05-20 18:46:02 UTC` (`2026-05-20 15:46:02 America/Sao_Paulo`).

### Current to-do list

1. Keep watching the active k2/k4 official run until the first four seeds finish.
   - Current active W&B IDs: `uk3ogrvq`, `22b516km`, `op06nvh0`, `5cwkp6o9`.
   - These are not done yet, but they are close enough to wait rather than intervene.
   - Once they finish, append the final rewards/action stds and let the remaining 6 k2/k4 trials start.

2. Let the currently active k8 init_std workers drain.
   - `linop8_std0p5` seed `395483`, temporary log `8668304487.log`, started but no W&B URL emitted yet.
   - `linop8_std1p0` seed `105487`, temporary log `4375547604.log`, started but no W&B URL emitted yet.
   - Do not count these as durable until their W&B run IDs appear.

3. Keep the k8 LR ablation queued.
   - `linop8_lr_ablation_std1p0` has no observed `linop8_lr0p...` worker logs or W&B IDs yet.
   - It is launched but resource-starved behind the active k2/k4 and init_std workers.
   - Next check should specifically look for `Run name: linop8_lr0p` in `/tmp/isaacray_trials/*.log`.

4. After the first resource wave finishes, decide whether to wait for all init_std pending trials or prioritize LR ablation.
   - Evidence so far says init_std is not a clean k=8 fix.
   - The k2/k4 run is the paper-facing priority.
   - The LR ablation is the mechanistic follow-up requested by the coworker.

5. If no LR workers have started after k2/k4 frees GPUs, check the LR tuner process before relaunching.
   - Tuner process exists for sweep `6baa2c8a`.
   - Do not relaunch unless the tuner has exited or errors; otherwise duplicate runs would pollute the official 5-seed record.

### Live metrics snapshot

Main k2/k4 official run:

| Variant | Seed | W&B ID | Temporary log | Latest iter | Latest reward | Action std | Episode length | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mlp3_linop2` | `182236` | `uk3ogrvq` | `2233736496.log` | 1200 | 20.94 | 0.38 | 1000.00 | active, strong |
| `mlp3_linop2` | `55898` | `22b516km` | `3225767421.log` | 1195 | 20.78 | 0.37 | 995.39 | active, strong |
| `mlp3_linop4` | `837442` | `op06nvh0` | `1715168587.log` | 1169 | 18.58 | 0.48 | 996.77 | active, stable |
| `mlp3_linop4` | `952264` | `5cwkp6o9` | `7467068784.log` | 1124 | 18.10 | 0.48 | 984.85 | active, stable |

k8 init_std active/recent state:

| Variant | Seed | W&B ID | Temporary log | Latest iter | Latest reward | Action std | Episode length | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `linop8_std1p0` | `181372` | `9dh8vkn2` | `5045786537.log` | 570 | -1.47 | 0.97 | 23.44 | active, stuck |
| `linop8_std0p3` | `680414` | `9d8os113` | `7012937996.log` | 549 | -1.43 | 0.31 | 16.32 | active, stuck |
| `linop8_std0p5` | `395483` | pending | `8668304487.log` | tuner reports ~348 | pending in log | pending | pending | active, no W&B yet |
| `linop8_std1p0` | `105487` | pending | `4375547604.log` | tuner reports ~244 | pending in log | pending | pending | active, no W&B yet |
| `linop8_std0p5` | `425776` | `yu25o6v6` | `9249468088.log` | 1500 | -2.41 | 0.48 | 864.79 | complete, poor despite long episodes |
| `linop8_std0p3` | `290447` | `0m210exq` | `2544665969.log` | 1500 | 3.36 | 0.31 | 870.78 | complete, partial recovery |

Interpretation at this checkpoint:

- k=4 is doing what we needed: it trains stably and should give a cleaner paper-consistent LinOp result than k=8.
- k=8 init_std remains inconsistent. Lower `init_std` can recover some seeds, but several seeds remain stuck or weak.
- The LR ablation is still the important missing mechanistic test. No data yet because it has not obtained workers.

---

## 2026-05-20 19:56 UTC live status

Snapshot time: `2026-05-20 19:56:38 UTC` (`2026-05-20 16:56:38 America/Sao_Paulo`).

### What changed since 18:46 UTC

- The first k2/k4 wave finished and the next k2/k4 wave is running.
- The remaining k8 init_std pending seeds started and now have durable W&B IDs.
- The k8 LR ablation still has no observed `linop8_lr0p...` worker logs or W&B IDs.

### New durable IDs

Main k2/k4 official run, newly observed workers:

| Experiment | Variant | k | Seed | W&B run ID | W&B URL | Temporary log | Latest iter | Latest reward | Action std | Episode length | Status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `985491` | `riml769b` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/riml769b | `3600800732.log` | 1090 | 20.46 | 0.38 | 987.42 | active, strong |
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `219212` | `riwex9o7` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/riwex9o7 | `4432810355.log` | 1115 | 20.22 | 0.39 | 978.67 | active, strong |
| `linop_k2_k4_main` | `mlp3_linop2` | 2 | `628880` | `wglw6wsk` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/wglw6wsk | `9663116406.log` | 1190 | 20.76 | 0.38 | 990.56 | active, strong |
| `linop_k2_k4_main` | `mlp3_linop4` | 4 | `430948` | `c1gr0693` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/c1gr0693 | `2074123120.log` | 1054 | 16.92 | 0.50 | 965.02 | active, stable |

k8 init_std, newly observed durable IDs:

| Experiment | Variant | k | Seed | W&B run ID | W&B URL | Temporary log | Latest iter | Latest reward | Action std | Episode length | Status |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `linop8_init_std_self_contained` | `linop8_std0p5` | 8 | `257646` | `vr65xs0t` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/vr65xs0t | `7006756995.log` | 789 | -0.99 | 0.50 | 16.96 | active, stuck |
| `linop8_init_std_self_contained` | `linop8_std1p0` | 8 | `105487` | `h0xnjd18` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/h0xnjd18 | `4375547604.log` | 994 | -2.60 | 0.95 | 22.30 | active, stuck |
| `linop8_init_std_self_contained` | `linop8_std0p3` | 8 | `872230` | `62uq6oyh` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/62uq6oyh | `9096453678.log` | 249 | -1.25 | 0.31 | 13.23 | active, early/stuck |

### Current interpretation

- k=2 remains very strong across the official sweep.
- k=4 is still healthy but trails k=2 by roughly 2-4 reward points in the observed active seeds.
- k8 init_std looks increasingly negative as a robust fix: the remaining std1p0/std0p5/std0p3 seeds are all currently stuck.
- The main missing result is still the LR-scaling arm. It has not started workers yet.

### Next action

Let the current active workers drain. Once k2/k4 and/or init_std release GPUs, check immediately for LR worker start:

```bash
docker exec isaacray-training bash -c 'grep -aEl "Run name: linop8_lr0p" /tmp/isaacray_trials/*.log 2>/dev/null'
```

If no `linop8_lr0p...` logs appear after resources free, inspect whether the `linop8_lr_ablation_std1p0` tuner process is still alive before relaunching.

---

## 2026-05-20 20:08 UTC intervention

Snapshot time: `2026-05-20 20:08:46 UTC` (`2026-05-20 17:08:46 America/Sao_Paulo`).

Killed three stuck k8 init_std workers to free GPU capacity:

| Variant | Seed | W&B run ID | W&B URL | Temporary log | Last observed iter | Last reward | Action std | Episode length | Disposition |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `linop8_std1p0` | `105487` | `h0xnjd18` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/h0xnjd18 | `4375547604.log` | 1219 | -2.40 | 0.92 | 25.00 | manually killed, stuck |
| `linop8_std0p5` | `257646` | `vr65xs0t` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/vr65xs0t | `7006756995.log` | 1024 | -0.70 | 0.48 | 20.22 | manually killed, stuck/weak |
| `linop8_std0p3` | `872230` | `62uq6oyh` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/62uq6oyh | `9096453678.log` | 489 | -0.67 | 0.31 | 15.33 | manually killed, stuck/weak |

Post-kill process check showed those three `-rid` process trees cleared.

Important caveat: queued jobs immediately began filling freed resources. Active train workers after the kill included:

- `mlp3_linop2` seeds `219212`, `985491`
- `mlp3_linop4` seeds `430948`, `777358`
- short smoke workers from `anymal_rough_smoke` / Spot smoke

The LR ablation still had no observed `linop8_lr0p...` worker at this checkpoint.

---

## 2026-05-20 21:01 UTC LR ablation started

Snapshot time: `2026-05-20 21:01:17 UTC` (`2026-05-20 18:01:17 America/Sao_Paulo`).

The k8 LR-scaling ablation is now running and occupying the visible GPU workers.

Observed LR-ablation durable IDs:

| Experiment | Variant | k | lr | init_std | Seed | W&B run ID | W&B URL | Temporary log | Latest iter | Latest reward | Action std | Episode length | Status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `linop8_lr_ablation_std1p0` | `linop8_lr0p000075_std1p0` | 8 | `7.5e-5` | 1.0 | `881905` | `k0idawyu` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/k0idawyu | `5040800030.log` | 1500 | 16.07 | 0.56 | 987.12 | complete, strong |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p00015_std1p0` | 8 | `1.5e-4` | 1.0 | `782099` | `j1hqwtxd` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/j1hqwtxd | `4397957072.log` | 1344 | 14.73 | 0.60 | 991.75 | active, strong |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p0003_std1p0` | 8 | `3e-4` | 1.0 | `941490` | `v75x3wta` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/v75x3wta | `4262611115.log` | 179 | -1.11 | 0.97 | 15.74 | active, stuck early |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p00015_std1p0` | 8 | `1.5e-4` | 1.0 | `727311` | `4recetq9` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/4recetq9 | `8584734091.log` | 54 | -1.13 | 0.99 | 13.38 | active, early |
| `linop8_lr_ablation_std1p0` | `linop8_lr0p000075_std1p0` | 8 | `7.5e-5` | 1.0 | `904078` | `xo1tmbam` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/xo1tmbam | `6265300920.log` | 39 | -1.13 | 0.99 | 19.34 | active, early |

Immediate read:

- The LR-scaling hypothesis is now supported by at least one strong seed:
  `lr=7.5e-5`, seed `881905`, reached reward `16.07` with `init_std=1.0`.
- `lr=1.5e-4` also has a strong active seed at reward `14.73`.
- The baseline `lr=3e-4` arm is still showing the familiar stuck early signature.
- This is the cleanest evidence so far that k=8 can be fixed at the optimization level rather than by lowering exploration.

---

## 2026-05-21 00:51 UTC final LR-ablation results

Snapshot time: `2026-05-21 00:51:34 UTC` (`2026-05-20 21:51:34 America/Sao_Paulo`).

All `linop8_lr_ablation_std1p0` workers are finished. No active RSL-RL train or Ray tuner processes were observed before extracting this table.

### Per-seed final table

| LR | Variant | Seed | W&B run ID | W&B URL | Temporary log | Final iter | Final reward | Action std | Episode length | Status |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `7.5e-5` | `linop8_lr0p000075_std1p0` | `881905` | `k0idawyu` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/k0idawyu | `5040800030.log` | 1499/1500 | 16.07 | 0.56 | 987.12 | complete, strong |
| `7.5e-5` | `linop8_lr0p000075_std1p0` | `904078` | `xo1tmbam` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/xo1tmbam | `6265300920.log` | 1499/1500 | 16.49 | 0.56 | 991.25 | complete, strong |
| `7.5e-5` | `linop8_lr0p000075_std1p0` | `289390` | `38q4z7zb` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/38q4z7zb | `4005878431.log` | 1499/1500 | 16.48 | 0.56 | 990.30 | complete, strong |
| `7.5e-5` | `linop8_lr0p000075_std1p0` | `310664` | `52p2iyx8` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/52p2iyx8 | `2383722606.log` | 1499/1500 | 16.55 | 0.56 | 995.50 | complete, strong |
| `7.5e-5` | `linop8_lr0p000075_std1p0` | `825309` | `vvrygjah` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/vvrygjah | `1288519710.log` | 1499/1500 | 16.76 | 0.55 | 995.64 | complete, strong |
| `1.5e-4` | `linop8_lr0p00015_std1p0` | `782099` | `j1hqwtxd` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/j1hqwtxd | `4397957072.log` | 1499/1500 | 15.88 | 0.58 | 1000.00 | complete, strong |
| `1.5e-4` | `linop8_lr0p00015_std1p0` | `727311` | `4recetq9` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/4recetq9 | `8584734091.log` | 1079/1500 | 10.45 | 0.72 | 998.95 | runtime failure after recovery |
| `1.5e-4` | `linop8_lr0p00015_std1p0` | `734681` | `w8nkvnz2` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/w8nkvnz2 | `6365166814.log` | 1499/1500 | 10.88 | 0.67 | 976.68 | complete, moderate |
| `1.5e-4` | `linop8_lr0p00015_std1p0` | `539674` | `l9ezn9fp` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/l9ezn9fp | `4721859653.log` | 1499/1500 | 12.45 | 0.64 | 991.39 | complete, good |
| `1.5e-4` | `linop8_lr0p00015_std1p0` | `340130` | `5xeuyly8` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/5xeuyly8 | `9305328103.log` | 1499/1500 | 13.11 | 0.63 | 972.91 | complete, good |
| `3e-4` | `linop8_lr0p0003_std1p0` | `941490` | `v75x3wta` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/v75x3wta | `4262611115.log` | 1499/1500 | 12.85 | 0.64 | 969.44 | complete, recovered |
| `3e-4` | `linop8_lr0p0003_std1p0` | `446381` | `z7knsxae` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/z7knsxae | `2651464216.log` | 1499/1500 | -0.78 | 0.83 | 26.05 | complete, stuck |
| `3e-4` | `linop8_lr0p0003_std1p0` | `694605` | `7cwdipro` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/7cwdipro | `8393722053.log` | 1499/1500 | 7.39 | 0.73 | 950.11 | complete, partial |
| `3e-4` | `linop8_lr0p0003_std1p0` | `127889` | `yse5naky` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/yse5naky | `2708125015.log` | 1499/1500 | 8.60 | 0.71 | 970.76 | complete, partial |
| `3e-4` | `linop8_lr0p0003_std1p0` | `610348` | `yrg7yvqf` | https://wandb.ai/tommaselli/anymal_flat_v6/runs/yrg7yvqf | `5869218334.log` | 2/1500 | catastrophic negative | 1.00 | 67.54 | runtime failure |

Failure detail:

- `1.5e-4`, seed `727311`: `RuntimeError: normal expects all elements of std >= 0.0` after reaching reward `10.45`.
- `3e-4`, seed `610348`: catastrophic reward at iter 2 followed by the same std runtime failure.

### Aggregate readout

| LR | Completed/usable seeds | Mean final reward | Qualitative result |
|---:|---:|---:|---|
| `7.5e-5` | 5/5 | 16.47 | robust fix; all seeds strong |
| `1.5e-4` | 5/5 if counting failed-after-recovery seed, 4/5 strict completed | 12.55 including failed-after-recovery seed; 13.08 strict completed | viable but less robust than `7.5e-5` |
| `3e-4` | 4/5 strict completed, 3/5 useful | dominated by one crash and one stuck seed | unstable baseline; confirms original failure mode |

Conclusion:

- The coworker suggestion was right: k=8 is primarily an optimization-scale problem, not just an exploration problem.
- Scaling LR by `2/8` from the k=2 control (`3e-4 * 2/8 = 7.5e-5`) fixes k=8 robustly with `init_std=1.0`.
- This is cleaner than lowering `init_std`: the `init_std` sweep remained seed-fragile, while `lr=7.5e-5` produced 5/5 strong k=8 seeds.
- For the paper story, k=4 remains the paper-consistent sweet spot, while k=8 can be presented as a stress test that requires inverse-k LR scaling.
