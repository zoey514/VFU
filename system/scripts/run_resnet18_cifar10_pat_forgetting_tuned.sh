#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/VFU_data}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/VFU_results/resnet18_cifar10_pat_core}"

PYTHONPATH=${PYTHONPATH:-system} python -u system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --data_dir "$DATA_DIR" \
  --download \
  --dataset cifar10 \
  --model resnet18 \
  --device cuda \
  --num_clients 10 \
  --join_ratio 0.2 \
  --participation_mode force_target \
  --target_client 0 \
  --max_rounds 100 \
  --global_rounds 100 \
  --early_stop_patience 5 \
  --repair_rounds 100 \
  --repair_early_stop_patience 5 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 32 \
  --embedding_dim 128 \
  --split_mode pathological \
  --classes_per_client 2 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.2 \
  --auditfu_mcr_strength 1.2 \
  --auditfu_subspace_rank 20 \
  --disable_osd \
  --repair_strength 0.12 \
  --repair_kd_lambda 0.5 \
  --repair_kd_temp 2.0 \
  --repair_feat_lambda 1.5 \
  --repair_var_lambda 0.0 \
  --repair_prox_lambda 0.05 \
  --repair_subspace_lambda 0.0 \
  --min_pre_retain_acc 0.45 \
  --min_retain_score 0.9 \
  --task_auc_tolerance 0.05 \
  --retrain_rounds 100 \
  --retrain_join_ratio 1.0 \
  --retrain_encoder_epochs 2 \
  --retrain_lr_encoder 0.01 \
  --auditfu_log_dir "$LOG_DIR"
