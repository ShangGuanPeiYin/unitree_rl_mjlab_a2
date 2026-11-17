#!/usr/bin/env bash
# Default is preflight only. Main coordinator adds --execute after independent acceptance.
# --learning-rate asserts the unchanged initial recipe; restored actual Adam lr stays 0.00001.
set -euo pipefail
cd /home/xxs/dog/unitree_rl_mjlab_a2
exec env MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v06_c1.py \
  --mode train \
  --recipe 0.6-go2refs-c1 --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 --rollout 48 --epochs 5 --minibatches 4 \
  --learning-rate 0.0003 --gamma 0.99 --entropy-coef 0.003 \
  --initial-std 0.30 --std-min 0.05 --std-max 1.00 \
  --initial-vx-min 0.30 --initial-vx-max 0.60 \
  --final-vx-min 0.30 --final-vx-max 0.60 --vy 0.0 --wz 0.0 \
  --max-updates 36000 \
  --phase-budgets F0=1000,H=12000,D=9000,L=6000,V=4000,R=4000 \
  --save-interval 100 --diagnostic-interval 50 --gate-interval 100 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --source-checkpoint logs/a2_ideal_stairs_v06/train_seed42_trial02_resume_000400_q007/checkpoints/stop_004224.pt \
  --source-sha256 108e03dc64effa94d2824bbf36842aee8f9c5fd9c2122be391b337706f4a51b9 \
  --acceptance-file logs/a2_ideal_stairs_v06_c1/implementation/acceptance/INDEPENDENT_ACCEPTANCE.md \
  --c1-manifest logs/a2_ideal_stairs_v06_c1/implementation/C1_BRIDGE_MANIFEST.json \
  --output-dir logs/a2_ideal_stairs_v06_c1/continue_seed42_from004224_trial01 \
  "$@"
