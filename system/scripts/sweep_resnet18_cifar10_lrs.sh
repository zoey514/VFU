#!/usr/bin/env bash
set -euo pipefail

for lr_head in 0.05 0.1; do
  for lr_encoder in 0.005 0.01; do
    out_dir="results/resnet18_cifar10_lrhead${lr_head}_lrenc${lr_encoder}"
    PYTHONPATH=system python -B system/experiments/standalone_cifar10_fedrep_srauditfu.py \
      --model resnet18 \
      --device auto \
      --num_clients 10 \
      --join_ratio 0.4 \
      --force_target_participation \
      --global_rounds 50 \
      --repair_rounds 10 \
      --head_epochs 2 \
      --encoder_epochs 2 \
      --batch_size 64 \
      --embedding_dim 128 \
      --max_train_samples 0 \
      --max_test_samples 0 \
      --max_audit_batches 10 \
      --new_client_adapt_steps 10 \
      --alpha 0.3 \
      --lr_head "${lr_head}" \
      --lr_encoder "${lr_encoder}" \
      --auditfu_mask topk \
      --auditfu_topk_ratio 0.2 \
      --auditfu_subspace_rank 4 \
      --auditfu_lr_log_rank 256 \
      --repair_kd_lambda 1.0 \
      --repair_kd_temp 2.0 \
      --repair_feat_lambda 0.1 \
      --repair_coral_lambda 1.0 \
      --auditfu_log_dir "${out_dir}"
  done
done
