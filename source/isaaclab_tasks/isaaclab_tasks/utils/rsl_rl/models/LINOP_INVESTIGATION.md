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
- init_std fix sweep: `/home/ray/Repos/IsaacRay/ray/hyperparameter_tuning/tasks/anymal/linop8_init_std_cfg.py`
- LR scaling arm: `/home/ray/Repos/IsaacRay/ray/hyperparameter_tuning/tasks/anymal/linop8_lr_scale_cfg.py`
- Trial logs (in container): `/tmp/isaacray_trials/*.log`
- Sweep launcher: `python3 sweep.py launch …` from `/home/ray/Repos/IsaacRay/`
- MLflow: `http://localhost:5000`

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
