"""Experiment launcher for SR-AuditFU in PFLlib.

Place this file under PFLlib's ``system/experiments`` directory or run it from
the repository root after adding ``SRAuditFU`` to ``system/main.py``.  The
launcher keeps the command line explicit so experiment scripts can reproduce
the report's Month-1/2/3 milestones on CIFAR-10, CIFAR-100, TinyImageNet, or
FEMNIST with FedRep-style shared representations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run SR-AuditFU experiments on PFLlib.")
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--join_ratio", type=float, default=0.2)
    parser.add_argument("--global_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_learning_rate", type=float, default=0.01)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--auditfu_mask", choices=["full", "topk", "relative"], default="relative")
    parser.add_argument("--auditfu_subspace_rank", type=int, default=8)
    parser.add_argument("--auditfu_repair_rounds", type=int, default=20)
    parser.add_argument("--auditfu_decay", type=float, default=0.95)
    parser.add_argument("--auditfu_topk_ratio", type=float, default=0.2)
    parser.add_argument("--auditfu_relative_threshold", type=float, default=0.55)
    parser.add_argument("--auditfu_lambda_dir", type=float, default=0.1)
    parser.add_argument("--auditfu_lambda_kd", type=float, default=1.0)
    parser.add_argument("--auditfu_lambda_feat", type=float, default=0.1)
    parser.add_argument("--auditfu_lambda_var", type=float, default=0.1)
    parser.add_argument("--auditfu_kd_tau", type=float, default=2.0)
    parser.add_argument("--auditfu_direction_basis_path", default="")
    parser.add_argument("--auditfu_log_dir", default="results/sr_auditfu")
    parser.add_argument("--enable_target_subspace_projection", dest="enable_target_subspace_projection", action="store_true", default=True)
    parser.add_argument("--disable_target_subspace_projection", dest="enable_target_subspace_projection", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    system_dir = Path(__file__).resolve().parents[1]
    main_py = system_dir / "main.py"
    if not main_py.exists():
        raise SystemExit(f"PFLlib main.py not found at {main_py}")

    cmd = [
        sys.executable,
        str(main_py),
        "-data",
        args.dataset,
        "-m",
        args.model,
        "-algo",
        "SRAuditFU",
        "-nc",
        str(args.num_clients),
        "-jr",
        str(args.join_ratio),
        "-gr",
        str(args.global_rounds),
        "-ls",
        str(args.local_epochs),
        "-lbs",
        str(args.batch_size),
        "-lr",
        str(args.local_learning_rate),
        "-dev",
        args.device,
        "-t",
        str(args.times),
        "--target_client",
        str(args.target_client),
        "--partition",
        args.partition,
        "--alpha",
        str(args.alpha),
        "--auditfu_mask",
        args.auditfu_mask,
        "--auditfu_subspace_rank",
        str(args.auditfu_subspace_rank),
        "--auditfu_repair_rounds",
        str(args.auditfu_repair_rounds),
        "--auditfu_decay",
        str(args.auditfu_decay),
        "--auditfu_topk_ratio",
        str(args.auditfu_topk_ratio),
        "--auditfu_relative_threshold",
        str(args.auditfu_relative_threshold),
        "--auditfu_lambda_dir",
        str(args.auditfu_lambda_dir),
        "--auditfu_lambda_kd",
        str(args.auditfu_lambda_kd),
        "--auditfu_lambda_feat",
        str(args.auditfu_lambda_feat),
        "--auditfu_lambda_var",
        str(args.auditfu_lambda_var),
        "--auditfu_kd_tau",
        str(args.auditfu_kd_tau),
        "--auditfu_log_dir",
        args.auditfu_log_dir,
    ]
    if args.enable_target_subspace_projection:
        cmd.append("--enable_target_subspace_projection")
    else:
        cmd.append("--disable_target_subspace_projection")
    if args.auditfu_direction_basis_path:
        cmd.extend(["--auditfu_direction_basis_path", args.auditfu_direction_basis_path])
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=system_dir)


if __name__ == "__main__":
    main()
