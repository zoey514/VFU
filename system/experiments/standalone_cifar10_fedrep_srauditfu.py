"""Standalone CIFAR-10 FedRep + SR-AuditFU smoke/experiment runner.

This file does not require a full PFLlib checkout.  It is meant for quickly
running the method in the current workspace:

* shared CNN encoder / embedding layer is aggregated by the server;
* each client owns a private linear classification head;
* data are split with a Dirichlet label-skew partition;
* client updates are logged and then target-client unlearning is performed by
  MCR + target-subspace server projection repair.

The implementation is intentionally compact but keeps the same decomposition as
the report: shared representation is the unlearning object, local heads remain
personalized and are never aggregated.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from flcore.unlearning.auditfu import (
    AuditLogger,
    SRAuditConfig,
    apply_mask,
    build_target_subspace,
    deterministic_select_clients,
    estimate_contribution,
    estimate_total_abs_contribution,
    flatten_tensors,
    make_mask,
    mcr_remove_,
    project_orthogonal,
    representation_audit_scores,
    json_digest,
    orthogonal_steepest_descent_direction,
    project_against_direction,
    shared_state_dict,
    subtract_state_dict,
    tensor_digest,
    unlearning_cross_entropy,
    unflatten_vector,
)


class ConvEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        return self.embedding(z)


class ResNet18Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.backbone = models.resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(8, channels))
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.bottleneck = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.GroupNorm(8, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.backbone(x))


class FedRepModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 10,
        model_name: str = "small_cnn",
        in_channels: int = 3,
    ):
        super().__init__()
        if model_name == "small_cnn":
            self.base = ConvEncoder(embedding_dim, in_channels)
        elif model_name == "resnet18":
            self.base = ResNet18Encoder(embedding_dim, in_channels)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.base(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dirichlet_partition(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    min_size: int,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    num_classes = int(targets.max() + 1)

    while True:
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            class_indices = np.where(targets == c)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, part in enumerate(np.split(class_indices, split_points)):
                client_indices[client_id].extend(part.tolist())
        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            for x in client_indices:
                rng.shuffle(x)
            return client_indices


def pathological_partition(
    targets: Sequence[int],
    num_clients: int,
    classes_per_client: int,
    min_size: int,
    seed: int,
) -> List[List[int]]:
    """FedOSD-style pathological split: each client receives NC label classes."""

    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    num_classes = int(targets_arr.max() + 1)
    classes_per_client = max(1, min(int(classes_per_client), num_classes))
    for _ in range(100):
        class_pools = {}
        for c in range(num_classes):
            idx = np.where(targets_arr == c)[0]
            rng.shuffle(idx)
            class_pools[c] = list(idx)
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        class_order = list(range(num_classes))
        for cid in range(num_clients):
            chosen = [class_order[(cid * classes_per_client + offset) % num_classes] for offset in range(classes_per_client)]
            for c in chosen:
                take = max(1, len(class_pools[c]) // max(1, num_clients * classes_per_client // num_classes))
                client_indices[cid].extend(class_pools[c][:take])
                class_pools[c] = class_pools[c][take:]
        leftovers = [idx for pool in class_pools.values() for idx in pool]
        rng.shuffle(leftovers)
        for offset, idx in enumerate(leftovers):
            client_indices[offset % num_clients].append(idx)
        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            for idx in client_indices:
                rng.shuffle(idx)
            return client_indices
    raise RuntimeError("Could not build pathological split with requested min_size.")


def build_federated_partition(
    targets: Sequence[int],
    num_clients: int,
    split_mode: str,
    alpha: float,
    classes_per_client: int,
    min_size: int,
    seed: int,
) -> List[List[int]]:
    if split_mode == "dirichlet":
        return dirichlet_partition(targets, num_clients, alpha, min_size, seed)
    if split_mode == "pathological":
        return pathological_partition(targets, num_clients, classes_per_client, min_size, seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def load_dataset(data_dir: str, dataset_name: str, download: bool):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        in_channels, num_classes = 1, 10
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_tf = train_tf
        train_set = datasets.MNIST(data_dir, train=True, download=download, transform=train_tf)
        test_set = datasets.MNIST(data_dir, train=False, download=download, transform=test_tf)
    elif dataset_name in {"fashionmnist", "fashion-mnist", "fashion"}:
        in_channels, num_classes = 1, 10
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        test_tf = train_tf
        train_set = datasets.FashionMNIST(data_dir, train=True, download=download, transform=train_tf)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=download, transform=test_tf)
    elif dataset_name == "femnist":
        in_channels, num_classes = 1, 62
        # Torchvision does not ship LEAF FEMNIST's writer-natural split.  EMNIST
        # byclass provides the same character-label space as a runnable proxy;
        # federated client heterogeneity is still controlled by split_mode.
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1736,), (0.3248,))])
        test_tf = train_tf
        train_set = datasets.EMNIST(data_dir, split="byclass", train=True, download=download, transform=train_tf)
        test_set = datasets.EMNIST(data_dir, split="byclass", train=False, download=download, transform=test_tf)
    elif dataset_name == "cifar10":
        in_channels, num_classes = 3, 10
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_set = datasets.CIFAR10(data_dir, train=True, download=download, transform=train_tf)
        test_set = datasets.CIFAR10(data_dir, train=False, download=download, transform=test_tf)
    elif dataset_name == "cifar100":
        in_channels, num_classes = 3, 100
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        train_set = datasets.CIFAR100(data_dir, train=True, download=download, transform=train_tf)
        test_set = datasets.CIFAR100(data_dir, train=False, download=download, transform=test_tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return train_set, test_set, in_channels, num_classes


def maybe_cap_indices(indices: List[List[int]], max_total: int, seed: int) -> List[List[int]]:
    if max_total <= 0:
        return indices
    rng = np.random.default_rng(seed)
    capped = []
    per_client = max(1, max_total // len(indices))
    for client_indices in indices:
        take = min(per_client, len(client_indices))
        capped.append(rng.choice(client_indices, size=take, replace=False).tolist())
    return capped


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def coral_loss(current: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if current.shape[0] < 2 or teacher.shape[0] < 2:
        return current.new_tensor(0.0)
    return F.mse_loss(current.var(dim=0, unbiased=False), teacher.var(dim=0, unbiased=False))


def fedrep_batch_loss(
    model: FedRepModel,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher_model: FedRepModel | None = None,
    kd_lambda: float = 0.0,
    kd_temp: float = 2.0,
    feat_lambda: float = 0.0,
    coral_lambda: float = 1.0,
) -> torch.Tensor:
    features = model.base(x)
    logits = model.head(features)
    loss = F.cross_entropy(logits, y)
    if teacher_model is None or (kd_lambda <= 0.0 and feat_lambda <= 0.0):
        return loss

    with torch.no_grad():
        teacher_features = teacher_model.base(x)
        teacher_logits = teacher_model.head(teacher_features)

    if kd_lambda > 0.0:
        loss = loss + kd_lambda * (
            F.kl_div(
                F.log_softmax(logits / kd_temp, dim=1),
                F.softmax(teacher_logits / kd_temp, dim=1),
                reduction="batchmean",
            )
            * (kd_temp**2)
        )

    if feat_lambda > 0.0:
        mean_loss = F.mse_loss(features.mean(dim=0), teacher_features.mean(dim=0))
        cov_loss = coral_loss(features, teacher_features)
        loss = loss + feat_lambda * (mean_loss + coral_lambda * cov_loss)
    return loss


def train_one_client(
    model: FedRepModel,
    loader: DataLoader,
    device: torch.device,
    head_epochs: int,
    encoder_epochs: int,
    lr_head: float,
    lr_encoder: float,
    teacher_model: FedRepModel | None = None,
    kd_lambda: float = 0.0,
    kd_temp: float = 2.0,
    feat_lambda: float = 0.0,
    coral_lambda: float = 1.0,
) -> float:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    total_loss = 0.0
    total_batches = 0

    set_requires_grad(model.base, False)
    set_requires_grad(model.head, True)
    head_opt = torch.optim.SGD(model.head.parameters(), lr=lr_head, momentum=0.9)
    for _ in range(head_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            head_opt.zero_grad(set_to_none=True)
            loss = fedrep_batch_loss(
                model, x, y, teacher_model, kd_lambda, kd_temp, feat_lambda, coral_lambda
            )
            loss.backward()
            head_opt.step()
            total_loss += float(loss.item())
            total_batches += 1

    set_requires_grad(model.base, True)
    set_requires_grad(model.head, False)
    encoder_opt = torch.optim.SGD(model.base.parameters(), lr=lr_encoder, momentum=0.9)
    for _ in range(encoder_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            encoder_opt.zero_grad(set_to_none=True)
            loss = fedrep_batch_loss(
                model, x, y, teacher_model, kd_lambda, kd_temp, feat_lambda, coral_lambda
            )
            loss.backward()
            encoder_opt.step()
            total_loss += float(loss.item())
            total_batches += 1

    set_requires_grad(model.base, True)
    set_requires_grad(model.head, True)
    return total_loss / max(1, total_batches)


def train_repair_client(
    model: FedRepModel,
    teacher: FedRepModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_encoder: float,
    lr_head: float,
    lambda_kd: float,
    lambda_feat: float,
    lambda_var: float,
    kd_tau: float,
) -> float:
    """Retained-client repair with CE + logit KD + feature moment stability."""

    model.train()
    teacher = copy.deepcopy(teacher).to(device)
    teacher.eval()
    set_requires_grad(model.base, True)
    set_requires_grad(model.head, True)
    opt = torch.optim.SGD(
        [
            {"params": model.base.parameters(), "lr": lr_encoder},
            {"params": model.head.parameters(), "lr": lr_head},
        ],
        momentum=0.9,
    )
    total_loss = 0.0
    total_batches = 0
    tau = max(float(kd_tau), 1.0e-6)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_z = teacher.base(x)
                teacher_logits = teacher.head(teacher_z)
            z = model.base(x)
            logits = model.head(z)
            ce = F.cross_entropy(logits, y)
            kd = F.kl_div(
                F.log_softmax(logits / tau, dim=1),
                F.softmax(teacher_logits / tau, dim=1),
                reduction="batchmean",
            ) * (tau * tau)
            mean_loss = F.mse_loss(z.mean(dim=0), teacher_z.mean(dim=0))
            var_loss = F.mse_loss(z.var(dim=0, unbiased=False), teacher_z.var(dim=0, unbiased=False))
            loss = ce + lambda_kd * kd + lambda_feat * (mean_loss + lambda_var * var_loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(1, total_batches)


def apply_shared_update_(model: FedRepModel, update: Mapping[str, torch.Tensor], scale: float = 1.0) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, delta in update.items():
            params[name].add_(delta.to(params[name].device, params[name].dtype), alpha=scale)


def aggregate_updates(
    template: Mapping[str, torch.Tensor],
    updates: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    out = {name: torch.zeros_like(tensor) for name, tensor in template.items()}
    for update, weight in zip(updates, weights):
        for name in out.keys():
            out[name].add_(update[name].float(), alpha=float(weight))
    return out


def project_update(
    update: Mapping[str, torch.Tensor],
    template: Mapping[str, torch.Tensor],
    basis: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    if basis is None:
        return dict(update)
    names = sorted(template.keys())
    flat = flatten_tensors(update, names)
    projected = project_orthogonal(flat, basis.cpu())
    return unflatten_vector(projected, {name: template[name].cpu() for name in names})


def shared_gradient_vector(
    model: FedRepModel,
    head_state: Mapping[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    loss_kind: str,
    max_batches: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    work_model = copy.deepcopy(model).to(device)
    work_model.head.load_state_dict(head_state)
    work_model.train()
    work_model.zero_grad(set_to_none=True)
    total_loss = None
    batches = 0
    for x, y in loader:
        if batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = work_model(x)
        loss = unlearning_cross_entropy(logits, y) if loss_kind == "uce" else F.cross_entropy(logits, y)
        total_loss = loss if total_loss is None else total_loss + loss
        batches += 1
    if total_loss is None:
        template = shared_state_dict(model)
        zeros = {name: torch.zeros_like(tensor) for name, tensor in template.items()}
        return flatten_tensors(zeros), zeros
    (total_loss / max(1, batches)).backward()
    grads = {}
    for name, param in dict(work_model.named_parameters()).items():
        if name in shared_state_dict(work_model):
            grads[name] = torch.zeros_like(param.detach().cpu()) if param.grad is None else param.grad.detach().float().cpu()
    return flatten_tensors(grads), grads


def osd_unlearning_update(
    model: FedRepModel,
    client_heads: Sequence[Mapping[str, torch.Tensor]],
    train_loaders: Sequence[DataLoader],
    retained_clients: Sequence[int],
    target_client: int,
    device: torch.device,
    args: argparse.Namespace,
    use_osd: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    template = shared_state_dict(model)
    gu, _ = shared_gradient_vector(
        model,
        client_heads[target_client],
        train_loaders[target_client],
        device,
        "uce",
        args.osd_max_batches,
    )
    retained_grads = []
    selected_retained = deterministic_select_clients(
        retained_clients,
        min(args.osd_retain_clients, len(retained_clients)),
        {"seed": args.seed, "phase": "osd_retained_gradients", "target_client": target_client},
    )
    for cid in selected_retained:
        grad, _ = shared_gradient_vector(
            model,
            client_heads[cid],
            train_loaders[cid],
            device,
            "ce",
            args.osd_max_batches,
        )
        retained_grads.append(grad)
    G = torch.stack(retained_grads, dim=0) if retained_grads else None
    direction = orthogonal_steepest_descent_direction(gu, G) if use_osd else gu
    update = unflatten_vector(-args.osd_lr * direction.cpu(), template)
    metrics = {
        "target_grad_norm": float(torch.linalg.norm(gu).item()),
        "osd_direction_norm": float(torch.linalg.norm(direction).item()),
        "retained_gradient_count": int(0 if G is None else G.shape[0]),
        "used_osd": bool(use_osd),
    }
    return update, metrics


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney ROC AUC without sklearn."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end + 1) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = original_ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float()
    y = y.detach().float()
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return 0.0
    x = x[:n] - x[:n].mean(dim=0, keepdim=True)
    y = y[:n] - y[:n].mean(dim=0, keepdim=True)
    numerator = torch.linalg.norm(x.T.matmul(y), ord="fro").pow(2)
    denominator = torch.linalg.norm(x.T.matmul(x), ord="fro") * torch.linalg.norm(y.T.matmul(y), ord="fro")
    return float((numerator / (denominator + 1.0e-12)).item())


def flatten_state(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return flatten_tensors({name: tensor.detach().float().cpu() for name, tensor in state.items()})


def parameter_distance(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    l_vec = flatten_state(left)
    r_vec = flatten_state(right)
    diff = l_vec - r_vec
    return {
        "l2": float(torch.linalg.norm(diff).item()),
        "relative_l2": float(torch.linalg.norm(diff).item() / (torch.linalg.norm(r_vec).item() + 1.0e-12)),
        "cosine": float(F.cosine_similarity(l_vec.unsqueeze(0), r_vec.unsqueeze(0)).item()) if l_vec.numel() else 0.0,
    }


def tensor_dict_bytes(tensors: Mapping[str, torch.Tensor], dtype_bytes: int = 4) -> int:
    return int(sum(t.numel() * dtype_bytes for t in tensors.values()))


def mask_sparsity(mask: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    total = sum(t.numel() for t in mask.values())
    active = sum(int(t.detach().float().sum().item()) for t in mask.values())
    return {
        "active_parameters": int(active),
        "total_parameters": int(total),
        "active_ratio": float(active / max(1, total)),
        "sparsity": float(1.0 - active / max(1, total)),
    }


def select_training_clients(
    rng: random.Random,
    num_clients: int,
    join_clients: int,
    target_client: int,
    force_target: bool,
) -> List[int]:
    if not force_target:
        return sorted(rng.sample(range(num_clients), join_clients))
    if join_clients <= 1:
        return [target_client]
    pool = [cid for cid in range(num_clients) if cid != target_client]
    selected = rng.sample(pool, min(join_clients - 1, len(pool)))
    selected.append(target_client)
    return sorted(selected)


def audit_seed_payload(args: argparse.Namespace, phase: str, round_id: int, chain_hash: str) -> Dict[str, object]:
    return {
        "exp_seed": int(args.seed),
        "phase": phase,
        "round_id": int(round_id),
        "participation_mode": getattr(args, "participation_mode", "normal"),
        "target_client": int(args.target_client),
        "prev_chain_hash": chain_hash,
    }


def select_clients_for_training_round(
    args: argparse.Namespace,
    round_id: int,
    join_clients: int,
    chain_hash: str,
) -> List[int]:
    client_ids = list(range(args.num_clients))
    mode = args.participation_mode
    if mode == "normal":
        return deterministic_select_clients(
            client_ids,
            join_clients,
            audit_seed_payload(args, "train", round_id, chain_hash),
        )
    if mode == "force_target":
        return deterministic_select_clients(
            client_ids,
            join_clients,
            audit_seed_payload(args, "train", round_id, chain_hash),
            include_client=args.target_client,
        )
    if mode == "balanced_force_target":
        if join_clients <= 1:
            return [args.target_client]
        retained = [cid for cid in client_ids if cid != args.target_client]
        ranked = deterministic_select_clients(
            retained,
            len(retained),
            {"seed": args.seed, "phase": "balanced_force_target_order"},
        )
        take = min(join_clients - 1, len(ranked))
        start = (round_id * take) % max(1, len(ranked))
        selected = [ranked[(start + offset) % len(ranked)] for offset in range(take)]
        selected.append(args.target_client)
        return sorted(selected)
    raise ValueError(f"Unsupported participation_mode: {mode}")


def train_selected_clients_round(
    global_model: FedRepModel,
    client_heads: List[Mapping[str, torch.Tensor]],
    selected: Sequence[int],
    train_loaders: Sequence[DataLoader],
    train_indices: Sequence[Sequence[int]],
    device: torch.device,
    args: argparse.Namespace,
    teacher_model: FedRepModel | None = None,
    teacher_heads: Sequence[Mapping[str, torch.Tensor]] | None = None,
    kd_lambda: float = 0.0,
    kd_temp: float = 2.0,
    feat_lambda: float = 0.0,
    var_lambda: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], List[Mapping[str, torch.Tensor]], List[float], List[float]]:
    state_before = shared_state_dict(global_model)
    updates = []
    weights = []
    losses = []
    selected_sample_count = sum(len(train_indices[cid]) for cid in selected)
    for cid in selected:
        local_model = copy.deepcopy(global_model).to(device)
        local_model.head.load_state_dict(client_heads[cid])
        local_teacher = None
        if teacher_model is not None:
            local_teacher = copy.deepcopy(teacher_model).to(device)
            if teacher_heads is not None:
                local_teacher.head.load_state_dict(teacher_heads[cid])
        loss = train_one_client(
            local_model,
            train_loaders[cid],
            device,
            args.head_epochs,
            args.encoder_epochs,
            args.lr_head,
            args.lr_encoder,
            teacher_model=local_teacher,
            kd_lambda=kd_lambda,
            kd_temp=kd_temp,
            feat_lambda=feat_lambda,
            coral_lambda=var_lambda,
        )
        client_heads[cid] = copy.deepcopy(local_model.head.cpu().state_dict())
        local_model.to(device)
        update = subtract_state_dict(shared_state_dict(local_model), state_before)
        updates.append(update)
        weights.append(len(train_indices[cid]) / max(1, selected_sample_count))
        losses.append(loss)
    return aggregate_updates(state_before, updates, weights), updates, weights, losses


@torch.no_grad()
def evaluate_personalized(
    global_model: FedRepModel,
    client_heads: Sequence[Mapping[str, torch.Tensor]],
    loaders: Sequence[DataLoader],
    device: torch.device,
    target_client: int | None = None,
) -> Dict[str, float]:
    total_correct = 0
    total_seen = 0
    client_accs = []
    num_classes = int(global_model.head.out_features)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model = copy.deepcopy(global_model).to(device)
    model.eval()
    for client_id, (head_state, loader) in enumerate(zip(client_heads, loaders)):
        model.head.load_state_dict(head_state)
        correct = 0
        seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            seen += int(y.numel())
            for truth, guess in zip(y.cpu(), pred.cpu()):
                confusion[int(truth), int(guess)] += 1
        acc = correct / max(1, seen)
        client_accs.append(acc)
        total_correct += correct
        total_seen += seen
    tp = confusion.diag().float()
    precision = tp / confusion.sum(dim=0).clamp_min(1).float()
    recall = tp / confusion.sum(dim=1).clamp_min(1).float()
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    out = {
        "weighted_acc": total_correct / max(1, total_seen),
        "mean_client_acc": float(np.mean(client_accs)) if client_accs else 0.0,
        "std_client_acc": float(np.std(client_accs)) if client_accs else 0.0,
        "macro_f1": float(f1.mean().item()),
        "per_client_acc": [float(x) for x in client_accs],
    }
    if target_client is not None and 0 <= target_client < len(client_accs):
        retain_accs = [acc for cid, acc in enumerate(client_accs) if cid != target_client]
        out.update(
            {
                "target_client_acc": float(client_accs[target_client]),
                "retain_mean_acc": float(np.mean(retain_accs)) if retain_accs else 0.0,
                "retain_min_acc": float(np.min(retain_accs)) if retain_accs else 0.0,
                "retain_max_acc": float(np.max(retain_accs)) if retain_accs else 0.0,
            }
        )
    return out


@torch.no_grad()
def collect_losses_and_confidences(
    global_model: FedRepModel,
    head_state: Mapping[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[List[float], List[float]]:
    model = copy.deepcopy(global_model).to(device)
    model.head.load_state_dict(head_state)
    model.eval()
    losses: List[float] = []
    confidences: List[float] = []
    for batch_id, (x, y) in enumerate(loader):
        if batch_id >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        batch_losses = F.cross_entropy(logits, y, reduction="none")
        probs = F.softmax(logits, dim=1)
        losses.extend([float(v) for v in batch_losses.cpu()])
        confidences.extend([float(v) for v in probs.max(dim=1).values.cpu()])
    return losses, confidences


def adapt_new_client_head(
    global_model: FedRepModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
) -> Dict[str, float]:
    model = copy.deepcopy(global_model).to(device)
    set_requires_grad(model.base, False)
    set_requires_grad(model.head, True)
    opt = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=0.9)
    model.train()
    completed = 0
    for x, y in train_loader:
        if completed >= steps:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        completed += 1
    metrics = evaluate_personalized(model, [model.head.state_dict()], [test_loader], device)
    return {
        "new_client_head_steps": int(completed),
        "new_client_adapt_acc": float(metrics["weighted_acc"]),
        "new_client_adapt_macro_f1": float(metrics["macro_f1"]),
    }


def run_repair_procedure(
    label: str,
    initial_model: FedRepModel,
    initial_heads: Sequence[Mapping[str, torch.Tensor]],
    teacher_model: FedRepModel | None,
    teacher_heads: Sequence[Mapping[str, torch.Tensor]] | None,
    train_loaders: Sequence[DataLoader],
    test_loaders: Sequence[DataLoader],
    train_indices: Sequence[Sequence[int]],
    retained_clients: Sequence[int],
    target_client: int,
    target_basis: torch.Tensor | None,
    join_clients: int,
    args: argparse.Namespace,
    device: torch.device,
    use_projection: bool,
    use_teacher: bool,
    audit_logger: AuditLogger | None = None,
    phase_times: Dict[str, List[float] | float] | None = None,
    communication: Dict[str, int] | None = None,
    repair_rounds: int | None = None,
    recovery_direction: torch.Tensor | None = None,
) -> Tuple[FedRepModel, List[Mapping[str, torch.Tensor]], Dict[str, object]]:
    model = copy.deepcopy(initial_model).to(device)
    heads = copy.deepcopy(list(initial_heads))
    best_model = copy.deepcopy(model).to(device)
    best_heads = copy.deepcopy(heads)
    best_metrics = evaluate_personalized(best_model, best_heads, test_loaders, device, target_client)
    best_round = 0
    stale_rounds = 0
    history = []
    completed_rounds = 0

    total_repair_rounds = args.repair_rounds if repair_rounds is None else int(repair_rounds)
    for repair_id in range(max(0, total_repair_rounds)):
        t0 = time.time()
        selected = deterministic_select_clients(
            retained_clients,
            min(join_clients, len(retained_clients)),
            audit_seed_payload(
                args,
                f"{label}_repair",
                repair_id,
                audit_logger.chain_hash if audit_logger is not None else "0" * 64,
            ),
            exclude_clients=[target_client],
        )
        state_before = shared_state_dict(model)
        aggregate, client_updates, weights, losses = train_selected_clients_round(
            model,
            heads,
            selected,
            train_loaders,
            train_indices,
            device,
            args,
            teacher_model=teacher_model if use_teacher else None,
            teacher_heads=teacher_heads if use_teacher else None,
            kd_lambda=args.repair_kd_lambda if use_teacher else 0.0,
            kd_temp=args.repair_kd_temp,
            feat_lambda=args.repair_feat_lambda if use_teacher else 0.0,
            var_lambda=args.repair_var_lambda,
        )
        hashes = []
        update_bytes = 0
        if audit_logger is not None:
            for cid, weight, update in zip(selected, weights, client_updates):
                record = audit_logger.log_client_update(args.global_rounds + repair_id, cid, weight, update)
                hashes.append(record.update_hash)
                update_bytes += tensor_dict_bytes(update)
        else:
            update_bytes = tensor_dict_bytes(aggregate) * len(selected)

        if communication is not None:
            communication["repair_upload_bytes"] += update_bytes
            communication["repair_download_bytes"] += update_bytes

        if recovery_direction is not None:
            names = sorted(state_before.keys())
            flat_aggregate = flatten_tensors(aggregate, names)
            flat_aggregate = project_against_direction(flat_aggregate, recovery_direction.cpu())
            aggregate = unflatten_vector(flat_aggregate.cpu(), {name: state_before[name].cpu() for name in names})
        if use_projection:
            aggregate = project_update(aggregate, state_before, target_basis)
        apply_shared_update_(model, aggregate, scale=args.repair_strength)
        state_after = shared_state_dict(model)
        if audit_logger is not None:
            audit_logger.log_repair_round(repair_id, selected, state_before, state_after, hashes, target_basis)

        elapsed = time.time() - t0
        if phase_times is not None:
            phase_times["repair_rounds"].append(elapsed)
        metrics = evaluate_personalized(model, heads, test_loaders, device, target_client)
        completed_rounds += 1
        history.append(
            {
                "round": repair_id + 1,
                "selected_clients": list(map(int, selected)),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "weighted_acc": float(metrics["weighted_acc"]),
                "retain_mean_acc": float(metrics["retain_mean_acc"]),
                "target_client_acc": float(metrics.get("target_client_acc", 0.0)),
                "used_projection": bool(use_projection),
                "used_teacher": bool(use_teacher),
                "time_seconds": float(elapsed),
            }
        )
        print(
            f"{label} repair {repair_id + 1:03d}/{total_repair_rounds}: "
            f"loss={np.mean(losses):.4f}, weighted_acc={metrics['weighted_acc']:.4f}, "
            f"retain_acc={metrics['retain_mean_acc']:.4f}, best_retain={best_metrics['retain_mean_acc']:.4f}, "
            f"time={elapsed:.1f}s"
        )

        if metrics["retain_mean_acc"] > best_metrics["retain_mean_acc"] + args.repair_min_delta:
            best_metrics = metrics
            best_model = copy.deepcopy(model).to(device)
            best_heads = copy.deepcopy(heads)
            best_round = repair_id + 1
            stale_rounds = 0
        else:
            stale_rounds += 1
        if args.repair_early_stop_patience >= 0 and stale_rounds >= args.repair_early_stop_patience:
            print(f"{label} repair early stopped at round {repair_id + 1}; restoring best round {best_round}.")
            break

    final_metrics = evaluate_personalized(model, heads, test_loaders, device, target_client)
    return best_model, best_heads, {
        "history": history,
        "completed_rounds": int(completed_rounds),
        "best_round": int(best_round),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "restored_best_checkpoint": True,
        "used_projection": bool(use_projection),
        "used_teacher": bool(use_teacher),
    }


def run_training_baseline(
    label: str,
    model: FedRepModel,
    heads: Sequence[Mapping[str, torch.Tensor]],
    allowed_clients: Sequence[int],
    train_loaders: Sequence[DataLoader],
    test_loaders: Sequence[DataLoader],
    train_indices: Sequence[Sequence[int]],
    target_client: int,
    join_clients: int,
    rounds: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    baseline_model = copy.deepcopy(model).to(device)
    baseline_heads = copy.deepcopy(list(heads))
    history = []
    for round_id in range(max(0, rounds)):
        selected = deterministic_select_clients(
            allowed_clients,
            min(join_clients, len(allowed_clients)),
            audit_seed_payload(args, label, round_id, "0" * 64),
            exclude_clients=[],
        )
        aggregate, _, _, losses = train_selected_clients_round(
            baseline_model,
            baseline_heads,
            selected,
            train_loaders,
            train_indices,
            device,
            args,
        )
        apply_shared_update_(baseline_model, aggregate)
        metrics = evaluate_personalized(baseline_model, baseline_heads, test_loaders, device, target_client)
        history.append(
            {
                "round": round_id + 1,
                "selected_clients": list(map(int, selected)),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "retain_mean_acc": float(metrics.get("retain_mean_acc", 0.0)),
                "weighted_acc": float(metrics["weighted_acc"]),
            }
        )
    return {
        "rounds": int(max(0, rounds)),
        "metrics": evaluate_personalized(baseline_model, baseline_heads, test_loaders, device, target_client),
        "history": history,
    }


def verify_audit_chain(logger: AuditLogger) -> Dict[str, float | int | bool]:
    prev = "0" * 64
    ok = True
    for record in logger.rounds:
        payload = {
            "round_id": record.round_id,
            "selected_clients": record.selected_clients,
            "state_hash_before": record.state_hash_before,
            "state_hash_after": record.state_hash_after,
            "update_root": record.update_root,
            "vrf_seed": record.vrf_seed,
            "prev_chain_hash": record.prev_chain_hash,
        }
        expected_chain = json_digest(payload)
        expected_seed = json_digest(
            {
                "round_id": record.round_id,
                "state_hash_before": record.state_hash_before,
                "prev_chain_hash": record.prev_chain_hash,
            }
        )
        ok = ok and record.prev_chain_hash == prev
        ok = ok and record.chain_hash == expected_chain
        ok = ok and record.vrf_seed == expected_seed
        prev = record.chain_hash
    repair_ok = True
    for record in logger.repair_rounds:
        payload = {
            "repair_id": record.repair_id,
            "selected_clients": record.selected_clients,
            "state_hash_before": record.state_hash_before,
            "state_hash_after": record.state_hash_after,
            "update_root": record.update_root,
            "vrf_seed": record.vrf_seed,
            "target_basis_hash": record.target_basis_hash,
            "prev_chain_hash": record.prev_chain_hash,
        }
        expected_chain = json_digest(payload)
        expected_seed = json_digest(
            {
                "phase": "repair",
                "repair_id": record.repair_id,
                "state_hash_before": record.state_hash_before,
                "target_basis_hash": record.target_basis_hash,
                "prev_chain_hash": record.prev_chain_hash,
            }
        )
        repair_ok = repair_ok and record.prev_chain_hash == prev
        repair_ok = repair_ok and record.chain_hash == expected_chain
        repair_ok = repair_ok and record.vrf_seed == expected_seed
        prev = record.chain_hash
    ok = ok and repair_ok
    return {
        "exec_audit_pass": bool(ok),
        "round_records": int(len(logger.rounds)),
        "repair_round_records": int(len(logger.repair_rounds)),
        "client_update_records": int(len(logger.client_updates)),
        "chain_verification_rate": float(1.0 if ok else 0.0),
        "merkle_root_presence_rate": float(
            np.mean([1.0 if r.update_root and len(r.update_root) == 64 else 0.0 for r in logger.rounds])
        )
        if logger.rounds
        else 0.0,
        "vrf_seed_presence_rate": float(
            np.mean([1.0 if r.vrf_seed and len(r.vrf_seed) == 64 else 0.0 for r in logger.rounds])
        )
        if logger.rounds
        else 0.0,
        "repair_vrf_seed_presence_rate": float(
            np.mean([1.0 if r.vrf_seed and len(r.vrf_seed) == 64 else 0.0 for r in logger.repair_rounds])
        )
        if logger.repair_rounds
        else 0.0,
    }


def task_inference_auc(
    target_embeddings: torch.Tensor,
    retain_embeddings: torch.Tensor,
    target_reference_embeddings: torch.Tensor,
    lambda_white: float,
) -> Dict[str, float]:
    if target_embeddings.numel() == 0 or retain_embeddings.numel() == 0:
        return {"auc_mahalanobis": 0.5, "auc_variance_proxy": 0.5, "auc_inner_proxy": 0.5}
    ref = target_reference_embeddings.detach().float()
    target = target_embeddings.detach().float()
    retain = retain_embeddings.detach().float()
    ref_mean = ref.mean(dim=0)
    centered = ref - ref_mean
    cov = centered.T.matmul(centered) / max(1, centered.shape[0] - 1)
    reg = lambda_white * torch.trace(cov) / max(1, cov.shape[0])
    cov = cov + reg * torch.eye(cov.shape[0], dtype=cov.dtype)
    inv_cov = torch.linalg.pinv(cov)

    all_embeddings = torch.cat([target, retain], dim=0)
    diffs = all_embeddings - ref_mean
    dists = torch.sum(diffs.matmul(inv_cov) * diffs, dim=1)
    labels = [1] * target.shape[0] + [0] * retain.shape[0]
    mahal_scores = [-float(x) for x in dists]

    coordinate_scores = [float(x.var().item()) for x in all_embeddings]
    normed = F.normalize(all_embeddings, dim=1)
    centroid = F.normalize(ref_mean.unsqueeze(0), dim=1).squeeze(0)
    inner_scores = [float(x.dot(centroid).abs().item()) for x in normed]
    return {
        "auc_mahalanobis": binary_auc(labels, mahal_scores),
        "auc_variance_proxy": binary_auc(labels, coordinate_scores),
        "auc_inner_proxy": binary_auc(labels, inner_scores),
    }


def confidence_interval_membership(value: float, out_values: Sequence[float]) -> Dict[str, float | bool]:
    arr = np.asarray(list(out_values), dtype=np.float64)
    if arr.size == 0:
        return {"within_95ci": False, "z_score": float("inf"), "out_mean": 0.0, "out_std": 0.0}
    mu = float(arr.mean())
    sigma = float(arr.std() + 1.0e-12)
    z = abs(value - mu) / sigma
    return {
        "within_95ci": bool(z <= 1.96),
        "z_score": float(z),
        "out_mean": mu,
        "out_std": sigma,
    }


@torch.no_grad()
def collect_embeddings(
    model: FedRepModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> torch.Tensor:
    model.eval()
    embeddings = []
    for batch_id, (x, _) in enumerate(loader):
        if batch_id >= max_batches:
            break
        embeddings.append(model.base(x.to(device)).detach().cpu())
    if not embeddings:
        return torch.empty(0, model.head.in_features)
    return torch.cat(embeddings, dim=0)


def concat_embeddings(
    model: FedRepModel,
    loaders: Sequence[DataLoader],
    device: torch.device,
    max_batches: int,
    exclude_client: int | None = None,
) -> torch.Tensor:
    chunks = []
    for cid, loader in enumerate(loaders):
        if exclude_client is not None and cid == exclude_client:
            continue
        chunks.append(collect_embeddings(model, loader, device, max_batches))
    chunks = [x for x in chunks if x.numel() > 0]
    if not chunks:
        return torch.empty(0, model.head.in_features)
    return torch.cat(chunks, dim=0)


def write_flat_metrics_csv(path: Path, metrics: Mapping[str, object]) -> None:
    rows: List[Tuple[str, object]] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            rows.append((prefix, json.dumps(value)))
        else:
            rows.append((prefix, value))

    visit("", metrics)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone CIFAR-10 FedRep + SR-AuditFU runner.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100", "femnist"],
        default="cifar10",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--join_ratio", type=float, default=0.1)
    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--total_rounds", type=int, default=100)
    parser.add_argument("--repair_rounds", type=int, default=20)
    parser.add_argument("--model", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--head_epochs", type=int, default=1)
    parser.add_argument("--encoder_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_head", type=float, default=0.05)
    parser.add_argument("--lr_encoder", type=float, default=0.05)
    parser.add_argument("--split_mode", choices=["pathological", "dirichlet"], default="pathological")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--classes_per_client", "--NC", dest="classes_per_client", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--target_client", type=int, default=0)
    parser.add_argument(
        "--participation_mode",
        choices=["normal", "force_target", "balanced_force_target"],
        default="normal",
        help="normal is the main experiment; force_target modes are stress tests.",
    )
    parser.add_argument("--force_target_participation", dest="force_target_participation", action="store_true", default=False)
    parser.add_argument("--no_force_target_participation", dest="force_target_participation", action="store_false")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--max_audit_batches", type=int, default=10)
    parser.add_argument("--new_client_adapt_steps", type=int, default=5)
    parser.add_argument("--osd_lr", type=float, default=0.0004)
    parser.add_argument("--osd_max_batches", type=int, default=1)
    parser.add_argument("--osd_retain_clients", type=int, default=10)
    parser.add_argument("--enable_osd", dest="enable_osd", action="store_true", default=True)
    parser.add_argument("--disable_osd", dest="enable_osd", action="store_false")
    parser.add_argument("--repair_kd_lambda", type=float, default=0.5)
    parser.add_argument("--repair_kd_temp", type=float, default=2.0)
    parser.add_argument("--repair_feat_lambda", type=float, default=0.5)
    parser.add_argument("--repair_var_lambda", type=float, default=0.1)
    parser.add_argument("--repair_prox_lambda", type=float, default=0.01)
    parser.add_argument("--repair_subspace_lambda", type=float, default=1.0)
    parser.add_argument("--repair_coral_lambda", type=float, default=None)
    parser.add_argument("--repair_strength", type=float, default=0.2)
    parser.add_argument("--repair_early_stop_patience", type=int, default=2)
    parser.add_argument("--repair_min_delta", type=float, default=1.0e-4)
    parser.add_argument("--min_pre_retain_acc", type=float, default=0.45)
    parser.add_argument("--min_retain_score", type=float, default=0.9)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--baseline_rounds", type=int, default=-1)
    parser.add_argument("--skip_retrain_baseline", action="store_true")
    parser.add_argument("--retrain_rounds", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--auditfu_mask", choices=["full", "topk", "relative"], default="topk")
    parser.add_argument("--auditfu_decay", type=float, default=0.95)
    parser.add_argument("--auditfu_mcr_strength", type=float, default=1.0)
    parser.add_argument("--auditfu_topk_ratio", type=float, default=0.1)
    parser.add_argument("--auditfu_relative_threshold", type=float, default=0.55)
    parser.add_argument("--auditfu_subspace_rank", type=int, default=20)
    parser.add_argument("--auditfu_lr_log_rank", type=int, default=256)
    parser.add_argument("--auditfu_log_dir", default="results/standalone_cifar10_fedrep_srauditfu")
    args = parser.parse_args()
    if args.force_target_participation and args.participation_mode == "normal":
        args.participation_mode = "force_target"
    if args.repair_coral_lambda is not None:
        args.repair_var_lambda = args.repair_coral_lambda
    args.force_target_participation = args.participation_mode in {"force_target", "balanced_force_target"}
    args.auditfu_lambda_kd = args.repair_kd_lambda
    args.auditfu_kd_tau = args.repair_kd_temp
    args.auditfu_lambda_feat = args.repair_feat_lambda
    args.auditfu_lambda_var = args.repair_var_lambda

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_set, test_set, in_channels, num_classes = load_dataset(args.data_dir, args.dataset, args.download)
    train_targets = train_set.targets.tolist() if torch.is_tensor(train_set.targets) else list(train_set.targets)
    test_targets = test_set.targets.tolist() if torch.is_tensor(test_set.targets) else list(test_set.targets)

    train_indices = build_federated_partition(
        train_targets,
        args.num_clients,
        args.split_mode,
        args.alpha,
        args.classes_per_client,
        1,
        args.seed,
    )
    test_indices = build_federated_partition(
        test_targets,
        args.num_clients,
        args.split_mode,
        args.alpha,
        args.classes_per_client,
        1,
        args.seed + 1,
    )
    train_indices = maybe_cap_indices(train_indices, args.max_train_samples, args.seed + 2)
    test_indices = maybe_cap_indices(test_indices, args.max_test_samples, args.seed + 3)

    train_loaders = [
        DataLoader(Subset(train_set, idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        for idx in train_indices
    ]
    test_loaders = [
        DataLoader(Subset(test_set, idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for idx in test_indices
    ]

    global_model = FedRepModel(args.embedding_dim, num_classes, args.model, in_channels).to(device)
    client_heads = [copy.deepcopy(global_model.head.state_dict()) for _ in range(args.num_clients)]
    audit_cfg = SRAuditConfig.from_args(args, target_client=args.target_client)
    audit_logger = AuditLogger(audit_cfg)
    join_clients = max(1, int(round(args.num_clients * args.join_ratio)))
    phase_times = {"train_rounds": [], "repair_rounds": [], "mcr_seconds": 0.0, "audit_seconds": 0.0}
    communication = {"train_upload_bytes": 0, "train_download_bytes": 0, "repair_upload_bytes": 0, "repair_download_bytes": 0}

    print(
        f"SR-AuditFU-OSD setup: dataset={args.dataset}, split={args.split_mode}, "
        f"clients={args.num_clients}, join_clients={join_clients}, alpha={args.alpha}, "
        f"NC={args.classes_per_client}, target_client={args.target_client}"
    )
    print("Shared part: model.base / personalized part: model.head")

    for round_id in range(args.global_rounds):
        t0 = time.time()
        selected = select_clients_for_training_round(args, round_id, join_clients, audit_logger.chain_hash)
        state_before = shared_state_dict(global_model)
        updates = []
        weights = []
        hashes = []
        losses = []

        for cid in selected:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.head.load_state_dict(client_heads[cid])
            loss = train_one_client(
                local_model,
                train_loaders[cid],
                device,
                args.head_epochs,
                args.encoder_epochs,
                args.lr_head,
                args.lr_encoder,
            )
            client_heads[cid] = copy.deepcopy(local_model.head.cpu().state_dict())
            local_model.to(device)
            update = subtract_state_dict(shared_state_dict(local_model), state_before)
            weight = len(train_indices[cid]) / max(1, sum(len(train_indices[x]) for x in selected))
            record = audit_logger.log_client_update(round_id, cid, weight, update)
            update_bytes = tensor_dict_bytes(update)
            communication["train_upload_bytes"] += update_bytes
            communication["train_download_bytes"] += update_bytes
            updates.append(update)
            weights.append(weight)
            hashes.append(record.update_hash)
            losses.append(loss)

        aggregate = aggregate_updates(state_before, updates, weights)
        apply_shared_update_(global_model, aggregate)
        state_after = shared_state_dict(global_model)
        audit_logger.log_round(round_id, selected, state_before, state_after, hashes)
        phase_times["train_rounds"].append(time.time() - t0)
        metrics = evaluate_personalized(global_model, client_heads, test_loaders, device, args.target_client)
        print(
            f"Round {round_id + 1:03d}/{args.global_rounds}: "
            f"loss={np.mean(losses):.4f}, weighted_acc={metrics['weighted_acc']:.4f}, "
            f"retain_acc={metrics['retain_mean_acc']:.4f}, time={phase_times['train_rounds'][-1]:.1f}s"
        )

    pre_unlearn_model = copy.deepcopy(global_model).to(device)
    pre_unlearn_heads = copy.deepcopy(client_heads)
    pre_shared_state = shared_state_dict(pre_unlearn_model)
    pre_metrics = evaluate_personalized(pre_unlearn_model, pre_unlearn_heads, test_loaders, device, args.target_client)
    target_records = audit_logger.updates_for_client(args.target_client)
    current_round = max([r.round_id for r in audit_logger.client_updates], default=0)
    template = shared_state_dict(global_model)
    mcr_t0 = time.time()
    target_contribution = estimate_contribution(target_records, audit_cfg, template, current_round)
    total_abs = estimate_total_abs_contribution(audit_logger.client_updates, template, current_round, audit_cfg.decay)
    mask = make_mask(target_contribution, audit_cfg, total_abs)
    masked_contribution = apply_mask(target_contribution, mask)
    mcr_remove_(global_model, masked_contribution, audit_cfg)
    target_basis = build_target_subspace(target_records, template, audit_cfg.subspace_rank, mask=mask)
    phase_times["mcr_seconds"] = time.time() - mcr_t0
    mcr_model = copy.deepcopy(global_model).to(device)
    mcr_shared_state = shared_state_dict(mcr_model)
    mcr_metrics = evaluate_personalized(mcr_model, pre_unlearn_heads, test_loaders, device, args.target_client)
    retained_clients = [cid for cid in range(args.num_clients) if cid != args.target_client]
    osd_metrics = {"used_osd": False, "target_grad_norm": 0.0, "osd_direction_norm": 0.0, "retained_gradient_count": 0}
    osd_direction = None
    if args.enable_osd:
        osd_update, osd_metrics = osd_unlearning_update(
            global_model,
            client_heads,
            train_loaders,
            retained_clients,
            args.target_client,
            device,
            args,
            use_osd=True,
        )
        osd_direction = flatten_tensors({name: -tensor / max(args.osd_lr, 1.0e-12) for name, tensor in osd_update.items()})
        apply_shared_update_(global_model, osd_update)
    osd_model = copy.deepcopy(global_model).to(device)
    osd_shared_state = shared_state_dict(osd_model)
    osd_metrics["metrics"] = evaluate_personalized(osd_model, pre_unlearn_heads, test_loaders, device, args.target_client)
    print(
        f"Unlearn target client {args.target_client}: records={len(target_records)}, "
        f"basis_rank={0 if target_basis is None else target_basis.shape[1]}, "
        f"mask={audit_cfg.mask}, osd={args.enable_osd}"
    )

    global_model, client_heads, full_repair_info = run_repair_procedure(
        "SR-AuditFU-OSD",
        global_model,
        client_heads,
        pre_unlearn_model,
        pre_unlearn_heads,
        train_loaders,
        test_loaders,
        train_indices,
        retained_clients,
        args.target_client,
        target_basis,
        join_clients,
        args,
        device,
        use_projection=True,
        use_teacher=True,
        audit_logger=audit_logger,
        phase_times=phase_times,
        communication=communication,
        recovery_direction=osd_direction,
    )

    baseline_rounds = args.repair_rounds if args.baseline_rounds < 0 else args.baseline_rounds
    retrain_rounds = args.global_rounds if args.retrain_rounds < 0 else args.retrain_rounds
    baseline_metrics: Dict[str, object] = {
        "NoUnlearn": {
            "metrics": pre_metrics,
            "description": "Keep the pre-unlearning model unchanged.",
        },
        "MCR-only": {
            "metrics": mcr_metrics,
            "description": "Apply masked contribution removal without retained-client repair.",
        },
        "SR-AuditFU-OSD": {
            "metrics": full_repair_info["best_metrics"],
            "history": full_repair_info["history"],
            "best_round": full_repair_info["best_round"],
            "completed_rounds": full_repair_info["completed_rounds"],
            "osd": osd_metrics,
            "description": "MCR + UCE/OSD + recovery-direction projection + target-subspace repair.",
        },
    }
    if not args.skip_baselines:
        finetune_uce_model = copy.deepcopy(pre_unlearn_model).to(device)
        finetune_uce_heads = copy.deepcopy(pre_unlearn_heads)
        finetune_update, finetune_osd = osd_unlearning_update(
            finetune_uce_model,
            finetune_uce_heads,
            train_loaders,
            retained_clients,
            args.target_client,
            device,
            args,
            use_osd=False,
        )
        apply_shared_update_(finetune_uce_model, finetune_update)
        baseline_metrics["Fine-tune"] = {
            "metrics": evaluate_personalized(finetune_uce_model, finetune_uce_heads, test_loaders, device, args.target_client),
            "osd": finetune_osd,
            "description": "FedOSD-style UCE target-client update without conflict mitigation or repair.",
        }

        mcr_repair_model, mcr_repair_heads, mcr_repair_info = run_repair_procedure(
            "MCR+Repair",
            mcr_model,
            pre_unlearn_heads,
            pre_unlearn_model,
            pre_unlearn_heads,
            train_loaders,
            test_loaders,
            train_indices,
            retained_clients,
            args.target_client,
            None,
            join_clients,
            args,
            device,
            use_projection=False,
            use_teacher=True,
            repair_rounds=baseline_rounds,
        )
        baseline_metrics["MCR+Repair"] = {
            "metrics": mcr_repair_info["best_metrics"],
            "history": mcr_repair_info["history"],
            "best_round": mcr_repair_info["best_round"],
            "completed_rounds": mcr_repair_info["completed_rounds"],
            "description": "MCR-only followed by KD/feature repair, without FedOSD OSD or target-subspace projection.",
        }

        sr_auditfu_model, sr_auditfu_heads, sr_auditfu_info = run_repair_procedure(
            "SR-AuditFU",
            mcr_model,
            pre_unlearn_heads,
            pre_unlearn_model,
            pre_unlearn_heads,
            train_loaders,
            test_loaders,
            train_indices,
            retained_clients,
            args.target_client,
            target_basis,
            join_clients,
            args,
            device,
            use_projection=True,
            use_teacher=True,
            repair_rounds=baseline_rounds,
        )
        baseline_metrics["SR-AuditFU"] = {
            "metrics": sr_auditfu_info["best_metrics"],
            "history": sr_auditfu_info["history"],
            "best_round": sr_auditfu_info["best_round"],
            "completed_rounds": sr_auditfu_info["completed_rounds"],
            "description": "Original SR-AuditFU: MCR + target-subspace projected retained repair, without UCE/OSD.",
        }

        fedosd_model = copy.deepcopy(pre_unlearn_model).to(device)
        fedosd_heads = copy.deepcopy(pre_unlearn_heads)
        fedosd_update, fedosd_osd = osd_unlearning_update(
            fedosd_model,
            fedosd_heads,
            train_loaders,
            retained_clients,
            args.target_client,
            device,
            args,
            use_osd=True,
        )
        fedosd_direction = flatten_tensors({name: -tensor / max(args.osd_lr, 1.0e-12) for name, tensor in fedosd_update.items()})
        apply_shared_update_(fedosd_model, fedosd_update)
        fedosd_model, fedosd_heads, fedosd_info = run_repair_procedure(
            "FedOSD-Adapted",
            fedosd_model,
            fedosd_heads,
            None,
            None,
            train_loaders,
            test_loaders,
            train_indices,
            retained_clients,
            args.target_client,
            None,
            join_clients,
            args,
            device,
            use_projection=False,
            use_teacher=False,
            repair_rounds=baseline_rounds,
            recovery_direction=fedosd_direction,
        )
        baseline_metrics["FedOSD-Adapted"] = {
            "metrics": fedosd_info["best_metrics"],
            "history": fedosd_info["history"],
            "best_round": fedosd_info["best_round"],
            "completed_rounds": fedosd_info["completed_rounds"],
            "osd": fedosd_osd,
            "description": "FedOSD UCE/OSD adapted directly to the shared encoder, without MCR or target-subspace repair.",
        }

        if not args.skip_retrain_baseline:
            retrain_model = FedRepModel(args.embedding_dim, num_classes, args.model, in_channels).to(device)
            retrain_heads = [copy.deepcopy(retrain_model.head.state_dict()) for _ in range(args.num_clients)]
            baseline_metrics["Retrain"] = run_training_baseline(
                "Retrain",
                retrain_model,
                retrain_heads,
                retained_clients,
                train_loaders,
                test_loaders,
                train_indices,
                args.target_client,
                join_clients,
                retrain_rounds,
                args,
                device,
            )

    audit_t0 = time.time()
    post_shared_state = shared_state_dict(global_model)
    target_embeddings = collect_embeddings(global_model, test_loaders[args.target_client], device, args.max_audit_batches)
    reference_client = retained_clients[0]
    reference_embeddings = collect_embeddings(global_model, test_loaders[reference_client], device, args.max_audit_batches)
    audit_scores = representation_audit_scores(target_embeddings, reference_embeddings, audit_cfg.audit_lambda_white)

    pre_target_embeddings = collect_embeddings(
        pre_unlearn_model, test_loaders[args.target_client], device, args.max_audit_batches
    )
    post_target_embeddings = target_embeddings
    mcr_target_embeddings = collect_embeddings(mcr_model, test_loaders[args.target_client], device, args.max_audit_batches)
    pre_retain_embeddings = concat_embeddings(
        pre_unlearn_model, test_loaders, device, args.max_audit_batches, exclude_client=args.target_client
    )
    post_retain_embeddings = concat_embeddings(
        global_model, test_loaders, device, args.max_audit_batches, exclude_client=args.target_client
    )
    mcr_retain_embeddings = concat_embeddings(
        mcr_model, test_loaders, device, args.max_audit_batches, exclude_client=args.target_client
    )

    before_metrics = pre_metrics
    after_metrics = evaluate_personalized(global_model, client_heads, test_loaders, device, args.target_client)

    pre_mia_train_loss, pre_mia_train_conf = collect_losses_and_confidences(
        pre_unlearn_model, pre_unlearn_heads[args.target_client], train_loaders[args.target_client], device, args.max_audit_batches
    )
    pre_mia_test_loss, pre_mia_test_conf = collect_losses_and_confidences(
        pre_unlearn_model, pre_unlearn_heads[args.target_client], test_loaders[args.target_client], device, args.max_audit_batches
    )
    post_mia_train_loss, post_mia_train_conf = collect_losses_and_confidences(
        global_model, client_heads[args.target_client], train_loaders[args.target_client], device, args.max_audit_batches
    )
    post_mia_test_loss, post_mia_test_conf = collect_losses_and_confidences(
        global_model, client_heads[args.target_client], test_loaders[args.target_client], device, args.max_audit_batches
    )
    pre_mia_labels = [1] * len(pre_mia_train_loss) + [0] * len(pre_mia_test_loss)
    post_mia_labels = [1] * len(post_mia_train_loss) + [0] * len(post_mia_test_loss)

    retain_cka_values = []
    for cid in retained_clients:
        pre_z = collect_embeddings(pre_unlearn_model, test_loaders[cid], device, args.max_audit_batches)
        post_z = collect_embeddings(global_model, test_loaders[cid], device, args.max_audit_batches)
        retain_cka_values.append(linear_cka(pre_z, post_z))

    task_pre = task_inference_auc(
        pre_target_embeddings, pre_retain_embeddings, pre_target_embeddings, audit_cfg.audit_lambda_white
    )
    task_mcr = task_inference_auc(
        mcr_target_embeddings, mcr_retain_embeddings, pre_target_embeddings, audit_cfg.audit_lambda_white
    )
    task_post = task_inference_auc(
        post_target_embeddings, post_retain_embeddings, pre_target_embeddings, audit_cfg.audit_lambda_white
    )

    retain_out_scores = [
        representation_audit_scores(
            collect_embeddings(global_model, test_loaders[cid], device, args.max_audit_batches),
            reference_embeddings,
            audit_cfg.audit_lambda_white,
        )["pairwise_inner_abs_mean"]
        for cid in retained_clients
    ]
    target_ci = confidence_interval_membership(audit_scores["pairwise_inner_abs_mean"], retain_out_scores)

    new_client_id = retained_clients[-1]
    new_client_pre = adapt_new_client_head(
        pre_unlearn_model,
        train_loaders[new_client_id],
        test_loaders[new_client_id],
        device,
        args.new_client_adapt_steps,
        args.lr_head,
    )
    new_client_post = adapt_new_client_head(
        global_model,
        train_loaders[new_client_id],
        test_loaders[new_client_id],
        device,
        args.new_client_adapt_steps,
        args.lr_head,
    )

    utility_metrics = {
        "pre_unlearn": before_metrics,
        "after_mcr": mcr_metrics,
        "post_repair": after_metrics,
        "retain_accuracy_drop_from_pre": float(before_metrics["retain_mean_acc"] - after_metrics["retain_mean_acc"]),
        "macro_f1_drop_from_pre": float(before_metrics["macro_f1"] - after_metrics["macro_f1"]),
        "new_client_adaptation_pre": new_client_pre,
        "new_client_adaptation_post": new_client_post,
    }
    forgetting_metrics = {
        "task_inference_auc_pre": task_pre,
        "task_inference_auc_after_mcr": task_mcr,
        "task_inference_auc_post": task_post,
        "mia_auc_pre_loss": binary_auc(pre_mia_labels, [-x for x in pre_mia_train_loss + pre_mia_test_loss]),
        "mia_auc_pre_confidence": binary_auc(pre_mia_labels, pre_mia_train_conf + pre_mia_test_conf),
        "mia_auc_post_loss": binary_auc(post_mia_labels, [-x for x in post_mia_train_loss + post_mia_test_loss]),
        "mia_auc_post_confidence": binary_auc(post_mia_labels, post_mia_train_conf + post_mia_test_conf),
        "target_cka_pre_to_mcr": linear_cka(pre_target_embeddings, mcr_target_embeddings),
        "target_cka_pre_to_post": linear_cka(pre_target_embeddings, post_target_embeddings),
        "retain_cka_pre_to_post_mean": float(np.mean(retain_cka_values)) if retain_cka_values else 0.0,
        "retain_cka_pre_to_post_min": float(np.min(retain_cka_values)) if retain_cka_values else 0.0,
        "target_to_retain_cka_ratio": float(
            linear_cka(pre_target_embeddings, post_target_embeddings) / (np.mean(retain_cka_values) + 1.0e-12)
        )
        if retain_cka_values
        else 0.0,
        "param_distance_pre_to_mcr": parameter_distance(pre_shared_state, mcr_shared_state),
        "param_distance_pre_to_post": parameter_distance(pre_shared_state, post_shared_state),
        "param_distance_mcr_to_post": parameter_distance(mcr_shared_state, post_shared_state),
        "mask_sparsity": mask_sparsity(mask),
    }
    representation_metrics = {
        "post_target_vs_reference": audit_scores,
        "target_pairwise_inner_within_retain_95ci": target_ci,
        "target_embedding_samples": int(post_target_embeddings.shape[0]),
        "retain_embedding_samples": int(post_retain_embeddings.shape[0]),
    }
    execution_metrics = verify_audit_chain(audit_logger)
    fs_tia = 1.0 - min(1.0, 2.0 * abs(task_post["auc_mahalanobis"] - 0.5))
    fs_mia = 1.0 - min(1.0, 2.0 * abs(forgetting_metrics["mia_auc_post_loss"] - 0.5))
    fs_cka = 1.0 - min(1.0, max(0.0, forgetting_metrics["target_cka_pre_to_post"]))
    retain_score = min(
        1.0,
        after_metrics["retain_mean_acc"] / (before_metrics["retain_mean_acc"] + 1.0e-12),
    )
    raw_score_sr_auditfu = float(
        (1.0 if execution_metrics["exec_audit_pass"] else 0.0)
        * retain_score
        * (0.4 * fs_tia + 0.3 * fs_mia + 0.3 * fs_cka)
    )
    pre_utility_valid = bool(before_metrics["retain_mean_acc"] >= args.min_pre_retain_acc)
    repair_utility_valid = bool(retain_score >= args.min_retain_score)
    score_valid = bool(pre_utility_valid and repair_utility_valid and execution_metrics["exec_audit_pass"])
    invalid_reasons = []
    if not pre_utility_valid:
        invalid_reasons.append(
            f"pre retain accuracy {before_metrics['retain_mean_acc']:.4f} < min_pre_retain_acc {args.min_pre_retain_acc:.4f}"
        )
    if not repair_utility_valid:
        invalid_reasons.append(
            f"retain score {retain_score:.4f} < min_retain_score {args.min_retain_score:.4f}"
        )
    if not execution_metrics["exec_audit_pass"]:
        invalid_reasons.append("execution audit failed")
    audit_score_metrics = {
        "exec_gate": float(1.0 if execution_metrics["exec_audit_pass"] else 0.0),
        "score_valid": bool(score_valid),
        "invalid_reasons": invalid_reasons,
        "min_pre_retain_acc": float(args.min_pre_retain_acc),
        "min_retain_score": float(args.min_retain_score),
        "pre_utility_valid": bool(pre_utility_valid),
        "repair_utility_valid": bool(repair_utility_valid),
        "retain_score": float(retain_score),
        "forgetting_score_tia": float(fs_tia),
        "forgetting_score_mia": float(fs_mia),
        "forgetting_score_cka": float(fs_cka),
        "raw_score_sr_auditfu": float(raw_score_sr_auditfu),
        "score_sr_auditfu": float(raw_score_sr_auditfu if score_valid else 0.0),
    }
    shared_bytes = tensor_dict_bytes(template)
    basis_bytes = 0 if target_basis is None else int(target_basis.numel() * 4)
    storage_metrics = {
        "shared_parameters": int(sum(t.numel() for t in template.values())),
        "shared_model_bytes_fp32": int(shared_bytes),
        "full_history_update_bytes_fp32": int(sum(tensor_dict_bytes(r.shared_update) for r in audit_logger.client_updates)),
        "lr_log_bytes_fp32": int(
            sum(0 if r.low_rank_update is None else r.low_rank_update.numel() * 4 for r in audit_logger.client_updates)
        ),
        "target_subspace_bytes_fp32": basis_bytes,
        "audit_json_records": int(
            len(audit_logger.rounds) + len(audit_logger.repair_rounds) + len(audit_logger.client_updates)
        ),
    }
    system_cost_metrics = {
        "phase_times": {
            "train_rounds": phase_times["train_rounds"],
            "repair_rounds": phase_times["repair_rounds"],
            "train_total_seconds": float(sum(phase_times["train_rounds"])),
            "repair_total_seconds": float(sum(phase_times["repair_rounds"])),
            "mcr_seconds": float(phase_times["mcr_seconds"]),
            "audit_seconds": 0.0,
        },
        "communication": communication,
        "storage": storage_metrics,
    }
    phase_times["audit_seconds"] = time.time() - audit_t0
    system_cost_metrics["phase_times"]["audit_seconds"] = float(phase_times["audit_seconds"])

    threshold_checks = {
        "exec_audit_pass": bool(execution_metrics["exec_audit_pass"]),
        "pre_retain_acc_ge_min": bool(pre_utility_valid),
        "retain_score_ge_min": bool(repair_utility_valid),
        "score_valid": bool(score_valid),
        "task_inf_post_mahalanobis_le_0_55": bool(task_post["auc_mahalanobis"] <= 0.55),
        "mia_post_loss_le_0_55": bool(forgetting_metrics["mia_auc_post_loss"] <= 0.55),
        "retain_cka_ge_0_85": bool(forgetting_metrics["retain_cka_pre_to_post_mean"] >= 0.85),
        "target_cka_ratio_le_0_8": bool(forgetting_metrics["target_to_retain_cka_ratio"] <= 0.8),
        "target_inner_in_retain_95ci": bool(target_ci["within_95ci"]),
    }
    report_metrics = {
        "SR-AuditFU-OSD": {
            "R-Acc": float(after_metrics["retain_mean_acc"]),
            "ASR": float(after_metrics.get("target_client_acc", 0.0)),
            "MIA-AUC": float(forgetting_metrics["mia_auc_post_loss"]),
            "TIA-AUC": float(task_post["auc_mahalanobis"]),
            "CKA": float(forgetting_metrics["retain_cka_pre_to_post_mean"]),
            "Dist-to-theta_T": float(forgetting_metrics["param_distance_pre_to_post"]["relative_l2"]),
            "AuditPass": float(1.0 if execution_metrics["exec_audit_pass"] else 0.0),
        },
        "baselines": {
            name: {
                "R-Acc": float(payload["metrics"].get("retain_mean_acc", 0.0)),
                "ASR": float(payload["metrics"].get("target_client_acc", 0.0)),
            }
            for name, payload in baseline_metrics.items()
            if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping)
        },
    }
    all_metrics = {
        "utility": utility_metrics,
        "forgetting": forgetting_metrics,
        "representation": representation_metrics,
        "execution_audit": execution_metrics,
        "audit_score": audit_score_metrics,
        "report_metrics": report_metrics,
        "baselines": baseline_metrics,
        "system_cost": system_cost_metrics,
        "threshold_checks": threshold_checks,
        "metadata": {
            "dataset": args.dataset,
            "split_mode": args.split_mode,
            "classes_per_client": args.classes_per_client,
            "algorithm": "FedRep+SRAuditFU-OSD",
            "model": args.model,
            "target_client": args.target_client,
            "retain_clients": retained_clients,
            "target_record_count": len(target_records),
            "force_target_participation": bool(args.force_target_participation),
            "participation_mode": args.participation_mode,
            "is_stress_test": bool(args.participation_mode != "normal"),
            "basis_rank": 0 if target_basis is None else int(target_basis.shape[1]),
            "args": vars(args),
        },
    }

    out_dir = Path(args.auditfu_log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_logger.export(
        out_dir / "evidence.json",
        extra={
            "standalone_runner": True,
            "dataset": args.dataset,
            "split_mode": args.split_mode,
            "algorithm": "FedRep+SRAuditFU-OSD",
            "target_client": args.target_client,
            "target_record_count": len(target_records),
            "basis_rank": 0 if target_basis is None else int(target_basis.shape[1]),
            "metrics_file": "metrics.json",
            "flat_metrics_file": "metrics_flat.csv",
            "threshold_checks": threshold_checks,
            "audit_score": audit_score_metrics,
            "report_metrics": report_metrics,
            "baseline_names": list(baseline_metrics.keys()),
            "participation_mode": args.participation_mode,
            "is_stress_test": bool(args.participation_mode != "normal"),
            "args": vars(args),
        },
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    write_flat_metrics_csv(out_dir / "metrics_flat.csv", all_metrics)

    print("Final metrics:")
    print(
        json.dumps(
            {
                "utility": {
                    "retain_acc_pre": utility_metrics["pre_unlearn"]["retain_mean_acc"],
                    "retain_acc_post": utility_metrics["post_repair"]["retain_mean_acc"],
                    "macro_f1_post": utility_metrics["post_repair"]["macro_f1"],
                },
                "forgetting": {
                    "task_auc_post": forgetting_metrics["task_inference_auc_post"]["auc_mahalanobis"],
                    "mia_auc_post_loss": forgetting_metrics["mia_auc_post_loss"],
                    "target_cka_pre_to_post": forgetting_metrics["target_cka_pre_to_post"],
                    "retain_cka_pre_to_post_mean": forgetting_metrics["retain_cka_pre_to_post_mean"],
                },
                "execution_audit": execution_metrics,
                "audit_score": audit_score_metrics,
                "baselines": {
                    name: {
                        "retain_acc": payload["metrics"].get("retain_mean_acc", 0.0),
                        "weighted_acc": payload["metrics"].get("weighted_acc", 0.0),
                    }
                    for name, payload in baseline_metrics.items()
                    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping)
                },
                "system_cost": system_cost_metrics,
                "threshold_checks": threshold_checks,
            },
            indent=2,
        )
    )
    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
