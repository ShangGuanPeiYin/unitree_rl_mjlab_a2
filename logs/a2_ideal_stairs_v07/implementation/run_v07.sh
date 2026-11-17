#!/usr/bin/env bash
# Default is preflight only. Coordinator supplies --execute after exact independent acceptance.
# learning-rate=0.0003 asserts the unchanged recipe; actual restored Adam lr is 7.593750000000002e-05.
set -euo pipefail
cd /home/xxs/dog/unitree_rl_mjlab_a2
exec env MUJOCO_GL=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/xxs/anaconda3/envs/a2/bin/python scripts/run_a2_ideal_stairs_v07.py \
  --mode train \
  --recipe 0.7-d22-eval-regression-r1 --seed 42 --device cuda:0 \
  --num-envs 512 --eval-num-envs 8 --rollout 48 --epochs 5 --minibatches 4 \
  --learning-rate 0.0003 --gamma 0.99 --entropy-coef 0.003 \
  --initial-std 0.30 --std-min 0.05 --std-max 1.00 \
  --initial-vx-min 0.30 --initial-vx-max 0.60 \
  --final-vx-min 0.30 --final-vx-max 0.60 --vy 0.0 --wz 0.0 \
  --max-updates 48000 \
  --phase-budgets F0=1000,H=12000,D=21000,L=6000,V=4000,R=4000 \
  --save-interval 100 --diagnostic-interval 50 --gate-interval 100 \
  --init-checkpoint /home/xxs/dog/eth_a2_locomotion_controller/models/model_2000.pt \
  --init-sha256 8ee1ee1d3513e57c4348eb3b23977d21f380701fd17ac08c073f8a311a4d6722 \
  --source-checkpoint logs/a2_ideal_stairs_v06_c1/continue_seed42_from004224_trial01/checkpoints/stop_004817.pt \
  --source-sha256 a41963a8bf5e6f1d4810b0a0c912ac65747d61bd481ea3432ef638a99e076e64 \
  --acceptance-file logs/a2_ideal_stairs_v07/implementation/acceptance/INDEPENDENT_ACCEPTANCE.md \
  --v07-manifest logs/a2_ideal_stairs_v07/implementation/V07_BRIDGE_MANIFEST.json \
  --output-dir logs/a2_ideal_stairs_v07/continue_seed42_from004817_trial01 \
  "$@"
