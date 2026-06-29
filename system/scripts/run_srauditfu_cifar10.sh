#!/usr/bin/env bash
set -euo pipefail

python system/experiments/run_srauditfu.py \
  --dataset Cifar10 \
  --model cnn \
  --num_clients 20 \
  --join_ratio 0.2 \
  --global_rounds 200 \
  --local_epochs 5 \
  --batch_size 32 \
  --local_learning_rate 0.01 \
  --partition dirichlet \
  --alpha 0.1 \
  --target_client 0 \
  --auditfu_mask relative \
  --auditfu_subspace_rank 8 \
  --auditfu_repair_rounds 20 \
  --auditfu_decay 0.95 \
  --auditfu_log_dir results/sr_auditfu/cifar10_alpha0.1_target0
