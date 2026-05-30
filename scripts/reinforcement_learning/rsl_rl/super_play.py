# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted gait-telemetry rollout for a deployed RSL-RL policy.

Built on play.py.  Drives the trained policy through a fixed command profile
(stance -> forward 0.5 m/s -> backward 0.5 m/s) and logs the unified HDF5
telemetry schema (see IsaacRay/CLAUDE.md) for every parallel env.  The HDF5 is
the deliverable: it is self-contained so it can be post-processed elsewhere
alongside real-robot data (gait curves, swing/stance diagnostics, COT, ...).

This script does NOT modify play.py/train.py or any sweep config.

Example (inside the isaacray-training container, from /workspace/isaaclab):

  ./_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/super_play.py \\
      --task Isaac-Velocity-Flat-Spot-MLP-v0 \\
      --checkpoint /path/to/model_final.pt --headless --num_envs 64 \\
      --output logs/super_play/spot.h5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Scripted gait-telemetry rollout of an RSL-RL policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (robot instances) to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained Nucleus checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# -- super_play specific
parser.add_argument("--command_speed", type=float, default=0.5, help="Forward/backward |v_x| command (m/s).")
parser.add_argument("--stance_time", type=float, default=2.0, help="Standing phase duration (s).")
parser.add_argument("--forward_time", type=float, default=5.0, help="Forward-walking phase duration (s).")
parser.add_argument("--backward_time", type=float, default=5.0, help="Backward-walking phase duration (s).")
parser.add_argument("--output", type=str, default=None, help="Output HDF5 path (default logs/super_play/<task>_<ts>.h5).")
parser.add_argument("--foot_regex", type=str, default=None, help="Override the auto-detected foot-body regex.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout MP4.")
parser.add_argument("--video_length", type=int, default=None, help="Video length in steps (default: full rollout).")
parser.add_argument("--video_dir", type=str, default=None, help="Output directory for rollout MP4s.")
parser.add_argument("--camera_eye", type=str, default=None, help="Viewer camera eye as 'x,y,z'.")
parser.add_argument("--camera_lookat", type=str, default=None, help="Viewer camera look-at point as 'x,y,z'.")
parser.add_argument("--camera_resolution", type=str, default="1280,720", help="Viewer resolution as 'width,height'.")
parser.add_argument(
    "--nominal", action="store_true", default=False,
    help="Disable domain-randomization startup events (material/mass/com) for a clean single-robot match.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401

# MyRelic task registration (optional — only present when the snapshot has been
# docker cp'd into /workspace/myrelic, same pattern as train.py).
import os as _os, sys as _sys
_myrelic_path = "/workspace/myrelic/relic"
if _os.path.isdir(_myrelic_path) and _myrelic_path not in _sys.path:
    _sys.path.insert(0, _myrelic_path)
try:
    import relic.tasks  # noqa: F401  # registers Isaac-Spot-Locomotion-*-v0
except ImportError:
    pass

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

installed_version = metadata.version("rsl-rl-lib")

# ---------------------------------------------------------------------------
# Per-robot body/joint naming (verified against isaaclab_assets robot cfgs).
#   foot : contact/foot bodies     knee : knee joints
#   hip_y: thigh swing (flexion)   hip_x: abduction/adduction (lateral)
# ---------------------------------------------------------------------------
ROBOT_PROFILES = {
    "anymal": {"foot": r".*FOOT",       "knee": r".*KFE",   "hip_y": r".*HFE",        "hip_x": r".*HAA"},
    "spot":   {"foot": r".*_foot",      "knee": r".*_kn",   "hip_y": r".*_hy",        "hip_x": r".*_hx"},
    "h1":     {"foot": r".*ankle_link", "knee": r".*_knee", "hip_y": r".*_hip_pitch", "hip_x": r".*_hip_roll"},
}
# Foot regexes tried (in order) when the task name matches no known profile.
_FALLBACK_FOOT_REGEXES = [r".*foot.*", r".*FOOT", r".*_foot", r".*ankle.*", r".*toe.*"]


def _resolve_profile(task: str) -> dict:
    t = task.lower()
    for key, prof in ROBOT_PROFILES.items():
        if key in t:
            print(f"[super_play] matched robot profile '{key}' for task '{task}'.")
            return dict(prof)
    print(f"[super_play] no robot profile for '{task}'; foot auto-detect, gait groups best-effort.")
    return {"foot": None, "knee": None, "hip_y": None, "hip_x": None}


def _find_joint_group(robot, regex):
    """Return (ids, names) for a joint regex, or ([], []) if regex is None / no match."""
    if regex is None:
        return [], []
    try:
        ids, names = robot.find_joints(regex, preserve_order=False)
        return ids, names
    except Exception:
        return [], []


def _build_leg_groups(foot_names, knee, hip_y, hip_x):
    """Map each foot to its leg's knee/hip joint indices via the leg-token prefix.

    Leg token = part of the foot body name before the first '_' (e.g. spot
    'fl_foot'->'fl', anymal 'LF_FOOT'->'LF', h1 'left_ankle_link'->'left').
    """
    def _match(token, ids, names):
        for jid, jname in zip(ids, names):
            if jname.startswith(token):
                return jid
        return None

    groups = {}
    for col, fname in enumerate(foot_names):
        token = fname.split("_")[0]
        groups[token] = {
            "foot_col": col,
            "knee_idx": _match(token, *knee),
            "hip_y_idx": _match(token, *hip_y),
            "hip_x_idx": _match(token, *hip_x),
        }
    return groups


# ---------------------------------------------------------------------------
# Telemetry collector — env-axis extension of the CLAUDE.md SimTelemetryCollector.
# Each step appends arrays shaped (N, D); datasets become (T, N, D).
# ---------------------------------------------------------------------------
import h5py
import numpy as np


def _parse_tuple(value: str | None, length: int, cast=float):
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != length:
        raise ValueError(f"Expected {length} comma-separated values, got: {value}")
    return tuple(cast(p) for p in parts)


def _camera_defaults(task: str):
    task_l = task.lower()
    if "h1" in task_l:
        return (3.4, -4.0, 1.9), (0.0, 0.0, 0.95)
    return (2.7, -3.2, 1.25), (0.0, 0.0, 0.45)


def _configure_viewer(env_cfg, task: str):
    default_eye, default_lookat = _camera_defaults(task)
    eye = _parse_tuple(args_cli.camera_eye, 3, float) or default_eye
    lookat = _parse_tuple(args_cli.camera_lookat, 3, float) or default_lookat
    resolution = _parse_tuple(args_cli.camera_resolution, 2, int) or (1280, 720)
    if getattr(env_cfg, "viewer", None) is not None:
        env_cfg.viewer.eye = eye
        env_cfg.viewer.lookat = lookat
        env_cfg.viewer.resolution = resolution
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
    print(f"[super_play] viewer eye={eye}, lookat={lookat}, resolution={resolution}")


def _disable_debug_visuals(obj) -> int:
    """Disable config-level debug visual markers such as command arrows and terrain axes."""
    import dataclasses as _dc

    seen = set()

    def _walk(value) -> int:
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        count = 0
        if hasattr(value, "debug_vis"):
            try:
                if getattr(value, "debug_vis"):
                    count += 1
                setattr(value, "debug_vis", False)
            except Exception:
                pass

        if _dc.is_dataclass(value):
            for field in _dc.fields(value):
                try:
                    count += _walk(getattr(value, field.name))
                except Exception:
                    pass
        elif isinstance(value, dict):
            for item in value.values():
                count += _walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                count += _walk(item)
        return count

    return _walk(obj)


class SimTelemetryCollector:
    def __init__(self, filename, buffer_size=512):
        self.filename = filename
        self.buffer_size = buffer_size
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        # start fresh
        if os.path.exists(filename):
            os.remove(filename)
        self._buffers = {}
        self._count = 0

    def step(self, data_dict):
        for k, v in data_dict.items():
            self._buffers.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        self._count += 1
        if self._count >= self.buffer_size:
            self.flush()

    def flush(self):
        if self._count == 0:
            return
        with h5py.File(self.filename, "a") as f:
            for k, v in self._buffers.items():
                arr = np.stack(v, axis=0)  # (chunk_T, N, D)
                if k in f:
                    dset = f[k]
                    n0 = dset.shape[0]
                    dset.resize(n0 + arr.shape[0], axis=0)
                    dset[n0:] = arr
                else:
                    f.create_dataset(k, data=arr, maxshape=(None,) + arr.shape[1:], chunks=True, compression="gzip")
        self._buffers = {}
        self._count = 0

    def write_attrs(self, attrs: dict):
        self.flush()
        with h5py.File(self.filename, "a") as f:
            for k, v in attrs.items():
                f.attrs[k] = v

    def close(self):
        self.flush()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Scripted gait-telemetry rollout."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- resolve the checkpoint (same logic as play.py) ----
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[super_play] no pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # ---- clean-rollout cfg edits (so the scripted command survives & robots walk uninterrupted) ----
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    schedule_time = args_cli.stance_time + args_cli.forward_time + args_cli.backward_time
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = schedule_time + 5.0

    # Disable disturbance events (push & external forces) so the rollout is clean.
    _DISABLE_EVENTS = ("push_robot", "base_external_force_torque")
    if getattr(env_cfg, "events", None) is not None:
        for _ev in _DISABLE_EVENTS:
            if hasattr(env_cfg.events, _ev):
                setattr(env_cfg.events, _ev, None)
        if args_cli.nominal:
            for ev in ("physics_material", "add_base_mass", "base_com"):
                if hasattr(env_cfg.events, ev):
                    setattr(env_cfg.events, ev, None)

    # Null curriculum terms whose path addresses reference the events we just disabled.
    # Otherwise the curriculum manager walks env.events.push_robot.params → AttributeError on None.
    import dataclasses as _dc
    if getattr(env_cfg, "curriculum", None) is not None and _dc.is_dataclass(env_cfg.curriculum):
        _DISABLE_CURRICULUM_KW = ("push", "force", "torque", "external")
        for _f in _dc.fields(env_cfg.curriculum):
            if any(kw in _f.name for kw in _DISABLE_CURRICULUM_KW):
                setattr(env_cfg.curriculum, _f.name, None)
                print(f"[super_play] nulled curriculum term: {_f.name}")

    disabled_debug_visuals = _disable_debug_visuals(env_cfg)
    if disabled_debug_visuals:
        print(f"[super_play] disabled {disabled_debug_visuals} debug visualization config flag(s).")

    if args_cli.video:
        _configure_viewer(env_cfg, args_cli.task)

    # ---- create env ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ---- command schedule (steps) ----
    base_env_pre_wrap = env.unwrapped
    dt = base_env_pre_wrap.step_dt
    n_stance = int(round(args_cli.stance_time / dt))
    n_forward = int(round(args_cli.forward_time / dt))
    n_backward = int(round(args_cli.backward_time / dt))
    total_steps = n_stance + n_forward + n_backward
    speed = args_cli.command_speed

    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.abspath(os.path.join("logs", "super_play", "videos"))
        os.makedirs(video_dir, exist_ok=True)
        video_length = args_cli.video_length or total_steps
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
            "name_prefix": "super-play",
        }
        print(f"[super_play] recording video to: {video_dir} ({video_length} steps)")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[super_play] loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- robot / sensor handles + profile resolution ----
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact = base_env.scene["contact_forces"]
    device = base_env.device
    num_envs = base_env.num_envs

    prof = _resolve_profile(args_cli.task)
    foot_regex = args_cli.foot_regex or prof["foot"]

    # foot bodies on the contact sensor (for forces) and on the robot (for heights)
    foot_sensor_ids, foot_names = [], []
    candidate_regexes = [foot_regex] if foot_regex else list(_FALLBACK_FOOT_REGEXES)
    for rgx in candidate_regexes:
        try:
            ids, names = contact.find_bodies(rgx, preserve_order=False)
        except Exception:
            ids, names = [], []
        if ids:
            foot_sensor_ids, foot_names, foot_regex = ids, names, rgx
            break
    if not foot_sensor_ids:
        raise RuntimeError(f"[super_play] no foot bodies found (tried {candidate_regexes}).")
    foot_robot_ids, _ = robot.find_bodies(foot_regex, preserve_order=False)
    print(f"[super_play] {len(foot_names)} foot bodies: {', '.join(foot_names)}")

    foot_sensor_ids_t = torch.tensor(foot_sensor_ids, device=device)
    foot_robot_ids_t = torch.tensor(foot_robot_ids, device=device)

    # gait joint groups + per-leg mapping
    knee = _find_joint_group(robot, prof["knee"])
    hip_y = _find_joint_group(robot, prof["hip_y"])
    hip_x = _find_joint_group(robot, prof["hip_x"])
    leg_groups = _build_leg_groups(foot_names, knee, hip_y, hip_x)
    print(f"[super_play] leg groups: {leg_groups}")

    # ---- output + collector ----
    if args_cli.output:
        out_path = args_cli.output
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.abspath(os.path.join("logs", "super_play", f"{task_name}_{ts}.h5"))
    collector = SimTelemetryCollector(out_path)
    print(f"[super_play] writing telemetry to: {out_path}")

    def _phase_and_cmd(step):
        if step < n_stance:
            return 0, 0.0
        if step < n_stance + n_forward:
            return 1, speed
        return 2, -speed

    cmd_term = base_env.command_manager.get_term("base_velocity")
    g = 9.81

    obs = env.get_observations()
    import math as _math
    print(f"[super_play] rolling out {total_steps} steps ({schedule_time:.1f}s @ {1.0/dt:.0f} Hz), "
          f"{num_envs} envs.")

    for step in range(total_steps):
        phase_id, vx = _phase_and_cmd(step)
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0

        with torch.inference_mode():
            t0 = time.perf_counter()
            actions = policy(obs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_dt = time.perf_counter() - t0

            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

            rd = robot.data
            fz = contact.data.net_forces_w[:, foot_sensor_ids_t, 2]  # (N, n_feet)
            foot_h = rd.body_pos_w[:, foot_robot_ids_t, 2]           # (N, n_feet) ground z~0 (flat)
            sample = {
                "preprocessed_velocity_cmd": cmd_term.vel_command_b.detach().cpu().numpy(),
                "raw_base_linear_velocity": rd.root_lin_vel_b[:, :3].detach().cpu().numpy(),
                "raw_projected_gravity": rd.projected_gravity_b.detach().cpu().numpy(),
                "raw_joint_positions": rd.joint_pos.detach().cpu().numpy(),
                "raw_joint_velocities": rd.joint_vel.detach().cpu().numpy(),
                "raw_joint_loads": rd.applied_torque.detach().cpu().numpy(),
                "dt_onnx_compute": np.full((num_envs, 1), infer_dt, dtype=np.float32),
                "contact_states": fz.gt(1.0).float().detach().cpu().numpy(),
                "raw_foot_forces_z": fz.detach().cpu().numpy(),
                "foot_heights": foot_h.detach().cpu().numpy(),
                "episode_dones": dones.float().reshape(num_envs, 1).detach().cpu().numpy(),
                "phase_id": np.full((num_envs, 1), phase_id, dtype=np.float32),
            }
        collector.step(sample)

    # ---- self-contained metadata ----
    import json
    robot_mass = rd.default_mass[0].sum().item() if rd.default_mass is not None else float("nan")
    per_env_mass = (
        rd.default_mass.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        if rd.default_mass is not None else np.full((num_envs,), robot_mass, dtype=np.float32)
    )
    collector.write_attrs({
        "robot_type": args_cli.task,
        "policy_id": os.path.basename(resume_path),
        "checkpoint_path": resume_path,
        "dt": float(dt),
        "g": float(g),
        "num_envs": int(num_envs),
        "command_speed": float(speed),
        "phase_boundaries": np.array([n_stance, n_stance + n_forward, total_steps], dtype=np.int64),
        "phase_labels": json.dumps({"0": "standing", "1": "forward", "2": "backward"}),
        "joint_names": json.dumps(list(robot.data.joint_names)),
        "foot_names": json.dumps(list(foot_names)),
        "foot_regex": foot_regex,
        "default_joint_pos": robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32),
        "robot_mass": per_env_mass,
        "leg_groups": json.dumps(leg_groups),
        "gait_regexes": json.dumps({k: prof[k] for k in ("knee", "hip_y", "hip_x")}),
    })
    collector.close()
    print(f"[super_play] done. {total_steps} steps written for {num_envs} envs -> {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
