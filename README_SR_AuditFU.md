# SR-AuditFU PFLlib Experiment Files

This workspace contains PFLlib-compatible experiment code for the report's
shared-representation personalized federated unlearning method.

## Files

- `system/flcore/unlearning/auditfu.py`: core SR-AuditFU primitives: audit logs,
  LR-Log projection, deterministic auditable scheduling, MCR,
  full/top-k/relative masks, masked target-subspace SVD, server projection,
  repair-round records, and representation audit scores.
- `system/flcore/clients/clientauditfu.py`: FedRep-style client with
  adversarial confusion, DV mutual-information minimization, feature moment
  stabilization, and proximal repair loss.
- `system/flcore/servers/serverauditfu.py`: PFLlib server entry point
  `SRAuditFU`, reusing FedRep/FedPer shared encoder training and adding
  auditable client unlearning.
- `system/experiments/run_srauditfu.py`: reproducible experiment launcher.
- `system/scripts/run_srauditfu_cifar10.sh`: CIFAR-10 Dirichlet non-IID example.

## PFLlib Integration

Copy these files into a PFLlib checkout, then add the following branch in
`system/main.py` where other algorithms are selected:

```python
from flcore.servers.serverauditfu import SRAuditFU

if args.algorithm == "SRAuditFU":
    server = SRAuditFU(args, i)
```

Add these argparse options to PFLlib's parser if it does not pass unknown
arguments through:

```python
parser.add_argument("--target_client", type=int, default=0)
parser.add_argument("--auditfu_mask", type=str, default="topk",
                    choices=["full", "topk", "relative"])
parser.add_argument("--auditfu_decay", type=float, default=0.95)
parser.add_argument("--auditfu_mcr_strength", type=float, default=1.0)
parser.add_argument("--auditfu_topk_ratio", type=float, default=0.2)
parser.add_argument("--auditfu_relative_threshold", type=float, default=0.55)
parser.add_argument("--auditfu_subspace_rank", type=int, default=8)
parser.add_argument("--auditfu_lr_log_rank", type=int, default=512)
parser.add_argument("--auditfu_repair_rounds", type=int, default=20)
parser.add_argument("--auditfu_lambda_adv", type=float, default=0.2)
parser.add_argument("--auditfu_lambda_mi", type=float, default=0.01)
parser.add_argument("--auditfu_lambda_prox", type=float, default=1e-4)
parser.add_argument("--auditfu_lambda_dir", type=float, default=0.1)
parser.add_argument("--auditfu_lambda_kd", type=float, default=1.0)
parser.add_argument("--auditfu_lambda_feat", type=float, default=0.1)
parser.add_argument("--auditfu_lambda_var", type=float, default=0.1)
parser.add_argument("--auditfu_kd_tau", type=float, default=2.0)
parser.add_argument("--auditfu_direction_basis_path", type=str, default="")
parser.add_argument("--auditfu_log_dir", type=str, default="results/sr_auditfu")
```

## Example

```bash
bash system/scripts/run_srauditfu_cifar10.sh
```

The run trains FedRep-style shared representations, logs client contributions,
removes `target_client` with MCR, performs retained-client repair with server
projection, and writes an evidence package under `auditfu_log_dir`.

## Standalone CIFAR-10 Smoke Run

The current workspace can also run without a full PFLlib checkout:

```bash
PYTHONPATH=system python system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --num_clients 5 \
  --join_ratio 0.4 \
  --global_rounds 2 \
  --repair_rounds 1 \
  --max_train_samples 1000 \
  --max_test_samples 500
```

This uses `model.base` as the shared embedding encoder and keeps one private
`model.head` classifier per client.

Add `--download` on the first run if CIFAR-10 is not already present under
`data/`. In restricted local sandboxes, keep `--num_workers 0` or omit it; this
is the default.

## Multi-Dimensional Evaluation

The standalone runner writes three files under `--auditfu_log_dir`:

- `metrics.json`: nested metrics grouped by evaluation dimension.
- `metrics_flat.csv`: one metric per row for tables and plotting.
- `evidence.json`: audit chain metadata and experiment provenance.

The metric groups follow the research report:

- `utility`: weighted accuracy, mean client accuracy, retain-client accuracy,
  target-client accuracy, macro-F1, per-client accuracy, and new-client head
  adaptation.
- `forgetting`: task-inference AUC, loss/confidence MIA AUC, target CKA,
  retain CKA, target/retain CKA ratio, parameter distances, and mask sparsity.
- `representation`: coordinate variance, pairwise inner product, Mahalanobis
  score, CKA-to-reference, and a retain-distribution 95% interval check.
- `execution_audit`: hash-chain verification, Merkle-root presence, VRF-seed
  presence, train/repair round counts, and update-record count.
- `audit_score`: the report's combined SR-AuditFU score from execution audit,
  retained utility, task-inference AUC, MIA AUC, and target CKA.
- `system_cost`: train/repair/MCR/audit timing, upload/download estimates, full
  history storage, LR-log storage, and target-subspace storage.

Minimal local validation command:

```bash
PYTHONPATH=system python -B system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --device cpu \
  --num_clients 3 \
  --join_ratio 0.67 \
  --global_rounds 1 \
  --repair_rounds 1 \
  --embedding_dim 16 \
  --max_train_samples 120 \
  --max_test_samples 90 \
  --max_audit_batches 1 \
  --new_client_adapt_steps 1 \
  --auditfu_log_dir results/smoke_multimetric_cifar10
```

## ResNet-18 + Bottleneck + Repair KD

For stronger CIFAR-10 experiments, use the ResNet-18 shared encoder with an
embedding bottleneck and repair losses:

```bash
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
  --embedding_dim 128 \
  --max_train_samples 0 \
  --max_test_samples 0 \
  --lr_head 0.05 \
  --lr_encoder 0.005 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.2 \
  --repair_kd_lambda 1.0 \
  --repair_kd_temp 2.0 \
  --repair_feat_lambda 0.1 \
  --repair_coral_lambda 1.0
```

The repair objective is now:

```text
CE(retain labels) + lambda_kd * KL(pre logits || post logits)
                 + lambda_feat * (embedding mean alignment + coordinate variance alignment)
```

Run the requested learning-rate sweep with:

```bash
bash system/scripts/sweep_resnet18_cifar10_lrs.sh
```
