#!/usr/bin/env bash
# Drive every robot's deployed policy through the scripted stance->forward->backward
# command profile and dump self-contained gait telemetry HDF5s.
#
# Run inside the isaacray-training container, from /workspace/isaaclab:
#   docker exec -it isaacray-training bash -lc \
#     '/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/launch_super_play.sh'
#
# Edit the CKPT_* paths to point at each robot's deployed checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> /workspace/isaaclab

PY=./_isaac_sim/python.sh
SP=scripts/reinforcement_learning/rsl_rl/super_play.py
OUTDIR=logs/super_play
COMMON="--headless --num_envs 64 --command_speed 0.5 --stance_time 2 --forward_time 5 --backward_time 5"

CKPT_SPOT=${CKPT_SPOT:-/path/to/spot/model_final.pt}
CKPT_ANYMAL=${CKPT_ANYMAL:-/path/to/anymal/model_final.pt}
CKPT_H1=${CKPT_H1:-/path/to/h1/model_final.pt}

$PY $SP --task Isaac-Velocity-Flat-Spot-MLP-v0   --checkpoint "$CKPT_SPOT"   $COMMON --output $OUTDIR/spot.h5
$PY $SP --task Isaac-Velocity-Flat-Anymal-D-v0   --checkpoint "$CKPT_ANYMAL" $COMMON --output $OUTDIR/anymal.h5
$PY $SP --task Isaac-Velocity-Flat-H1-MLP-v0     --checkpoint "$CKPT_H1"     $COMMON --output $OUTDIR/h1.h5

echo "Done. Telemetry HDF5s in $OUTDIR/. Post-process with super_play_metrics.py."
