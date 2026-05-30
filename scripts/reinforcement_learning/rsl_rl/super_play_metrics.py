# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline gait/locomotion metrics for super_play.py telemetry.

Pure numpy/h5py/scipy — no Isaac Sim, no plotting.  Reads a self-contained
super_play HDF5, segments the timeline by commanded v_x into standing / forward
/ backward phases (the same +/-0.1 m/s thresholds used on the real robot), and
computes the Section-2 core metrics and Section-4 swing/stance gait diagnostics.
Outputs a JSON; the real cross-comparison with hardware data happens elsewhere.

  python super_play_metrics.py file.h5 [--out metrics.json]

HDF5 datasets are (T, N, D); N = num parallel robots/envs.  Metrics are computed
per env (masking reset discontinuities) then aggregated mean/std across envs.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

V_THRESH = 0.1  # |v_cmd,x| <= 0.1 standing, > 0.1 forward, < -0.1 backward
HFR_CUTOFF_HZ = 10.0


def _agg(values: list[float]) -> dict:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def _phase_masks(vcmd_x: np.ndarray) -> dict:
    """vcmd_x: (T,) -> dict of boolean (T,) masks per phase."""
    return {
        "standing": np.abs(vcmd_x) <= V_THRESH,
        "forward": vcmd_x > V_THRESH,
        "backward": vcmd_x < -V_THRESH,
    }


def _hfr(q: np.ndarray, fs: float) -> float:
    """High-frequency ratio of multi-joint signal q (T, Ndof): power>cutoff / total."""
    if q.shape[0] < 4:
        return np.nan
    q = q - q.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(q.shape[0], d=1.0 / fs)
    power = np.abs(np.fft.rfft(q, axis=0)) ** 2  # (F, Ndof)
    total = power.sum()
    if total <= 0:
        return np.nan
    return float(power[freqs > HFR_CUTOFF_HZ].sum() / total)


def _stride_segments(contact_col: np.ndarray) -> list[tuple[int, int, int]]:
    """Takeoff->takeoff strides for one foot (Section-6 convention).

    A stride starts when the foot leaves the ground: swing first, then stance.
    contact_col: (T,) binary stance mask (1=stance). Returns (takeoff, touchdown,
    takeoff_next) so swing = [takeoff, touchdown), stance = [touchdown, takeoff_next).
    """
    c = contact_col.astype(np.int8)
    touchdowns = np.where((c[1:] == 1) & (c[:-1] == 0))[0] + 1  # 0->1 (land)
    takeoffs = np.where((c[1:] == 0) & (c[:-1] == 1))[0] + 1    # 1->0 (lift)
    strides = []
    for i in range(len(takeoffs) - 1):
        to_i, to_next = takeoffs[i], takeoffs[i + 1]
        td = touchdowns[(touchdowns > to_i) & (touchdowns < to_next)]
        if td.size:
            strides.append((int(to_i), int(td[0]), int(to_next)))
    return strides


def _resample(curve: np.ndarray, n: int = 101) -> np.ndarray:
    if curve.size < 2:
        return np.full(n, np.nan)
    x = np.linspace(0.0, 1.0, curve.size)
    return np.interp(np.linspace(0.0, 1.0, n), x, curve)


def compute(path: str) -> dict:
    with h5py.File(path, "r") as f:
        A = dict(f.attrs)
        vcmd = f["preprocessed_velocity_cmd"][:]        # (T,N,3)
        vbase = f["raw_base_linear_velocity"][:]        # (T,N,3)
        pgrav = f["raw_projected_gravity"][:]           # (T,N,3)
        qpos = f["raw_joint_positions"][:]              # (T,N,Ndof)
        qvel = f["raw_joint_velocities"][:]             # (T,N,Ndof)
        tau = f["raw_joint_loads"][:]                   # (T,N,Ndof)
        contact = f["contact_states"][:]                # (T,N,Nfeet)
        foot_h = f["foot_heights"][:]                   # (T,N,Nfeet)
        dones = f["episode_dones"][:, :, 0]             # (T,N)

    dt = float(A["dt"]); fs = 1.0 / dt; g = float(A["g"])
    default_q = np.asarray(A["default_joint_pos"], dtype=np.float64)  # (Ndof,)
    mass = np.asarray(A["robot_mass"], dtype=np.float64)              # (N,)
    leg_groups = json.loads(A["leg_groups"])
    T, N = qpos.shape[0], qpos.shape[1]

    # joint deltas relative to nominal standing pose (gait-write-up convention)
    dq = qpos - default_q[None, None, :]
    mech_power = np.abs(tau * qvel).sum(axis=2)  # (T,N) total |tau*dq|

    out = {
        "source": path,
        "robot_type": str(A.get("robot_type", "")),
        "policy_id": str(A.get("policy_id", "")),
        "dt": dt, "num_envs": int(N), "num_steps": int(T),
        "phases": {}, "gait": {}, "leg_groups": leg_groups,
    }

    # ---------- Section 2: core metrics, per phase ----------
    for phase in ("standing", "forward", "backward"):
        rmse, roll_std, pitch_std, hfr, cot = [], [], [], [], []
        for e in range(N):
            valid = dones[:, e] < 0.5
            m = _phase_masks(vcmd[:, e, 0])[phase] & valid
            if m.sum() < 4:
                continue
            # A. velocity tracking RMSE (xy)
            err = vcmd[m, e, :2] - vbase[m, e, :2]
            rmse.append(float(np.sqrt((err ** 2).sum(axis=1).mean())))
            # B. torso stability (attitude std)
            gx, gy, gz = pgrav[m, e, 0], pgrav[m, e, 1], pgrav[m, e, 2]
            pitch_std.append(float(np.std(np.arcsin(np.clip(gx, -1, 1)))))
            roll_std.append(float(np.std(np.arctan2(gy, -gz))))
            # C. control jitter HFR
            hfr.append(_hfr(qpos[m, e, :], fs))
            # D. cost of transport (moving phases only)
            if phase != "standing":
                vx = np.abs(vbase[m, e, 0]).mean()
                if vx > 1e-3:
                    cot.append(float(mech_power[m, e].mean() / (mass[e] * g * vx)))
        out["phases"][phase] = {
            "velocity_rmse": _agg(rmse),
            "pitch_std_rad": _agg(pitch_std),
            "roll_std_rad": _agg(roll_std),
            "hfr_jitter": _agg(hfr),
            "cost_of_transport": _agg(cot) if phase != "standing" else None,
        }

    # ---------- Section 4: swing/stance gait diagnostics (forward phase) ----------
    fwd = _phase_masks(vcmd[:, 0, 0])["forward"]  # command shared across envs
    fwd_idx = np.where(fwd)[0]
    for leg, grp in leg_groups.items():
        col = grp["foot_col"]
        knee_i, hipy_i, hipx_i = grp["knee_idx"], grp["hip_y_idx"], grp["hip_x_idx"]
        peak_flex, stance_std, hipx_range, stance_power, clearance, duty = [], [], [], [], [], []
        knee_curves, hipy_curves, hipx_curves = [], [], []
        for e in range(N):
            valid = dones[:, e] < 0.5
            seg_mask = fwd & valid
            if seg_mask.sum() < 4:
                continue
            strides = _stride_segments(contact[:, e, col])
            for to_i, td, to_next in strides:
                if not (seg_mask[to_i] and seg_mask[to_next - 1]):
                    continue
                sw = slice(to_i, td)       # swing (foot in air)
                st = slice(td, to_next)    # stance (foot on ground)
                full = slice(to_i, to_next)
                duty.append((to_next - td) / (to_next - to_i))  # stance fraction
                if td > to_i:
                    clearance.append(float(foot_h[sw, e, col].max()))
                if knee_i is not None:
                    k_sw = dq[sw, e, knee_i]; k_st = dq[st, e, knee_i]
                    if k_sw.size:
                        peak_flex.append(float(-k_sw.min()))  # magnitude of flexion
                        knee_curves.append(_resample(dq[full, e, knee_i]))
                    if k_st.size:
                        stance_std.append(float(np.std(k_st)))
                if hipx_i is not None:
                    hx = dq[full, e, hipx_i]
                    if hx.size:
                        hipx_range.append(float(hx.max() - hx.min()))
                        hipx_curves.append(_resample(hx))
                if hipy_i is not None and (to_next - to_i) > 1:
                    hipy_curves.append(_resample(dq[full, e, hipy_i]))
                st_p = mech_power[st, e]
                if st_p.size:
                    stance_power.append(float(st_p.mean()))

        def _mean_curve(curves):
            return np.nanmean(np.stack(curves), axis=0).tolist() if curves else None

        out["gait"][leg] = {
            "peak_swing_knee_flexion_rad": _agg(peak_flex),
            "stance_knee_stability_std_rad": _agg(stance_std),
            "lateral_hip_x_swing_rad": _agg(hipx_range),
            "avg_stance_mass_power_w": _agg(stance_power),
            "max_foot_clearance_m": _agg(clearance),
            "duty_factor": _agg(duty),
            "n_strides": len(duty),
            "mean_gait_curves_pct": {  # 0..100% stride progress, delta-from-nominal (rad)
                "knee": _mean_curve(knee_curves),
                "hip_y": _mean_curve(hipy_curves),
                "hip_x": _mean_curve(hipx_curves),
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Offline gait metrics for super_play HDF5.")
    ap.add_argument("h5", help="super_play telemetry .h5")
    ap.add_argument("--out", default=None, help="output JSON (default <h5>.metrics.json)")
    args = ap.parse_args()
    res = compute(args.h5)
    out = args.out or (args.h5.rsplit(".", 1)[0] + ".metrics.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[super_play_metrics] wrote {out}")
    # brief console summary
    for ph, d in res["phases"].items():
        rmse = d["velocity_rmse"]["mean"]
        print(f"  {ph:9s}  RMSE={rmse}")


if __name__ == "__main__":
    main()
