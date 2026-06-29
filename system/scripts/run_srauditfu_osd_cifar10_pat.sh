#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=system python -u system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --dataset cifar10 \
  --model small_cnn \
  --device auto \
  --num_clients 100 \
  --join_ratio 0.1 \
  --global_rounds 50 \
  --total_rounds 100 \
  --repair_rounds 20 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 200 \
  --lr_head 0.05 \
  --lr_encoder 0.05 \
  --split_mode pathological \
  --classes_per_client 2 \
  --target_client 0 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.1 \
  --auditfu_subspace_rank 20 \
  --osd_lr 0.0004 \
  --osd_max_batches 1 \
  --osd_retain_clients 10 \
  --repair_strength 0.2 \
  --repair_kd_lambda 0.5 \
  --repair_feat_lambda 0.5 \
  --repair_var_lambda 0.1 \
  --repair_prox_lambda 0.01 \
  --repair_subspace_lambda 1.0 \
  --repair_early_stop_patience 2 \
  --min_pre_retain_acc 0.45 \
  --min_retain_score 0.9 \
  --auditfu_log_dir results/srauditfu_osd_cifar10_pat_n100_nc2_c01
