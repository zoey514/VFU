"""Core utilities for SR-AuditFU experiments on top of shared-representation PFL.

The code is intentionally framework-light: the server/client wrappers call these
helpers from PFLlib, while the helpers themselves only assume PyTorch modules and
state_dict-style tensors.  SR-AuditFU follows the report design:

1. log every shared encoder update with hash-chain metadata;
2. locate the leaving client's historical contribution with time decay;
3. apply full, top-k, or relative masks for MCR;
4. build a target contribution subspace and project repair gradients;
5. export an auditable evidence package and black-box representation scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


DEFAULT_SHARED_KEYWORDS = (
    "base",
    "encoder",
    "feature",
    "features",
    "backbone",
    "representation",
    "body",
)
DEFAULT_HEAD_KEYWORDS = (
    "head",
    "classifier",
    "predictor",
    "projection_head",
    "fc",
)


def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu().contiguous()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    arr = _to_cpu_tensor(tensor).numpy()
    return sha256_bytes(arr.tobytes())


def json_digest(payload: Mapping) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(data)


def should_keep_shared_parameter(
    name: str,
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> bool:
    """Return whether a parameter belongs to the shared representation block."""

    low = name.lower()
    shared_keywords = tuple(shared_keywords or DEFAULT_SHARED_KEYWORDS)
    excluded_keywords = tuple(excluded_keywords or DEFAULT_HEAD_KEYWORDS)

    if any(token in low for token in excluded_keywords):
        return False
    if any(token in low for token in shared_keywords):
        return True
    # PFLlib split models usually expose model.base and model.head.  For custom
    # encoders without obvious names, use all non-head parameters as a fallback.
    return not any(token in low for token in excluded_keywords)


def named_shared_parameters(
    model: torch.nn.Module,
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> List[Tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
        and should_keep_shared_parameter(name, shared_keywords, excluded_keywords)
    ]


def shared_state_dict(
    model: torch.nn.Module,
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> TensorDict:
    return {
        name: _to_cpu_tensor(param)
        for name, param in named_shared_parameters(model, shared_keywords, excluded_keywords)
    }


def flatten_tensors(tensors: Mapping[str, torch.Tensor], names: Optional[Sequence[str]] = None) -> torch.Tensor:
    names = list(names or sorted(tensors.keys()))
    if not names:
        return torch.empty(0)
    return torch.cat([tensors[name].detach().float().reshape(-1).cpu() for name in names])


def unflatten_vector(vector: torch.Tensor, template: Mapping[str, torch.Tensor]) -> TensorDict:
    out: TensorDict = {}
    offset = 0
    for name in sorted(template.keys()):
        ref = template[name]
        size = ref.numel()
        out[name] = vector[offset : offset + size].view_as(ref).to(ref.device, ref.dtype)
        offset += size
    if offset != vector.numel():
        raise ValueError(f"Vector size mismatch: consumed {offset}, got {vector.numel()}.")
    return out


def subtract_state_dict(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> TensorDict:
    return {name: _to_cpu_tensor(left[name]) - _to_cpu_tensor(right[name]) for name in left.keys()}


def add_scaled_state_dict_(
    model: torch.nn.Module,
    update: Mapping[str, torch.Tensor],
    scale: float,
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> None:
    params = dict(named_shared_parameters(model, shared_keywords, excluded_keywords))
    with torch.no_grad():
        for name, delta in update.items():
            if name in params:
                params[name].add_(delta.to(params[name].device, params[name].dtype), alpha=scale)


@dataclass
class ClientUpdateRecord:
    round_id: int
    client_id: int
    aggregation_weight: float
    shared_update: TensorDict
    update_hash: str
    low_rank_update: Optional[torch.Tensor] = None

    def meta(self) -> Dict[str, object]:
        return {
            "round_id": self.round_id,
            "client_id": self.client_id,
            "aggregation_weight": self.aggregation_weight,
            "update_hash": self.update_hash,
            "has_low_rank_update": self.low_rank_update is not None,
        }


@dataclass
class AuditRoundRecord:
    round_id: int
    selected_clients: List[int]
    state_hash_before: str
    state_hash_after: str
    update_root: str
    vrf_seed: str
    prev_chain_hash: str
    chain_hash: str


@dataclass
class AuditRepairRecord:
    repair_id: int
    selected_clients: List[int]
    state_hash_before: str
    state_hash_after: str
    update_root: str
    vrf_seed: str
    target_basis_hash: str
    prev_chain_hash: str
    chain_hash: str


@dataclass
class SRAuditConfig:
    target_client: int
    decay: float = 0.95
    mcr_strength: float = 1.0
    mask: str = "topk"
    topk_ratio: float = 0.2
    relative_threshold: float = 0.55
    subspace_rank: int = 8
    lr_log_rank: int = 512
    projection_seed: int = 2026
    lambda_adv: float = 0.2
    lambda_mi: float = 0.01
    lambda_prox: float = 1.0e-4
    lambda_dir: float = 0.1
    lambda_kd: float = 1.0
    lambda_feat: float = 0.1
    lambda_var: float = 0.1
    kd_tau: float = 2.0
    direction_basis_path: str = ""
    audit_lambda_white: float = 0.1
    log_dir: str = "results/sr_auditfu"

    @classmethod
    def from_args(cls, args, target_client: Optional[int] = None) -> "SRAuditConfig":
        return cls(
            target_client=int(target_client if target_client is not None else getattr(args, "target_client", 0)),
            decay=float(getattr(args, "auditfu_decay", 0.95)),
            mcr_strength=float(getattr(args, "auditfu_mcr_strength", 1.0)),
            mask=str(getattr(args, "auditfu_mask", "topk")),
            topk_ratio=float(getattr(args, "auditfu_topk_ratio", 0.2)),
            relative_threshold=float(getattr(args, "auditfu_relative_threshold", 0.55)),
            subspace_rank=int(getattr(args, "auditfu_subspace_rank", 8)),
            lr_log_rank=int(getattr(args, "auditfu_lr_log_rank", 512)),
            projection_seed=int(getattr(args, "auditfu_projection_seed", 2026)),
            lambda_adv=float(getattr(args, "auditfu_lambda_adv", 0.2)),
            lambda_mi=float(getattr(args, "auditfu_lambda_mi", 0.01)),
            lambda_prox=float(getattr(args, "auditfu_lambda_prox", 1.0e-4)),
            lambda_dir=float(getattr(args, "auditfu_lambda_dir", 0.1)),
            lambda_kd=float(getattr(args, "auditfu_lambda_kd", 1.0)),
            lambda_feat=float(getattr(args, "auditfu_lambda_feat", 0.1)),
            lambda_var=float(getattr(args, "auditfu_lambda_var", 0.1)),
            kd_tau=float(getattr(args, "auditfu_kd_tau", 2.0)),
            direction_basis_path=str(getattr(args, "auditfu_direction_basis_path", "")),
            audit_lambda_white=float(getattr(args, "auditfu_audit_lambda_white", 0.1)),
            log_dir=str(getattr(args, "auditfu_log_dir", "results/sr_auditfu")),
        )


class LowRankProjector:
    """Deterministic Gaussian random projection for LR-Log."""

    def __init__(self, input_dim: int, rank: int, seed: int, device: torch.device = torch.device("cpu")):
        if rank <= 0:
            raise ValueError("rank must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        matrix = torch.randn(rank, input_dim, generator=generator, dtype=torch.float32)
        self.matrix = matrix / math.sqrt(rank)
        self.device = device

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matrix.to(vector.device).matmul(vector.detach().float().reshape(-1))


class AuditLogger:
    """Hash-chain metadata and update storage for client-level unlearning."""

    def __init__(self, config: SRAuditConfig):
        self.config = config
        self.rounds: List[AuditRoundRecord] = []
        self.repair_rounds: List[AuditRepairRecord] = []
        self.client_updates: List[ClientUpdateRecord] = []
        self.chain_hash = "0" * 64
        self._projector: Optional[LowRankProjector] = None

    def _state_hash(self, shared_state: Mapping[str, torch.Tensor]) -> str:
        payload = {name: tensor_digest(tensor) for name, tensor in sorted(shared_state.items())}
        return json_digest(payload)

    def _ensure_projector(self, vector: torch.Tensor) -> None:
        if self._projector is None:
            rank = min(self.config.lr_log_rank, max(1, vector.numel()))
            self._projector = LowRankProjector(vector.numel(), rank, self.config.projection_seed)

    def log_client_update(
        self,
        round_id: int,
        client_id: int,
        aggregation_weight: float,
        shared_update: Mapping[str, torch.Tensor],
    ) -> ClientUpdateRecord:
        flat = flatten_tensors(shared_update)
        self._ensure_projector(flat)
        update_hash = tensor_digest(flat)
        low_rank = self._projector.project(flat).cpu() if self._projector is not None else None
        record = ClientUpdateRecord(
            round_id=round_id,
            client_id=int(client_id),
            aggregation_weight=float(aggregation_weight),
            shared_update={name: _to_cpu_tensor(tensor) for name, tensor in shared_update.items()},
            update_hash=update_hash,
            low_rank_update=low_rank,
        )
        self.client_updates.append(record)
        return record

    def log_round(
        self,
        round_id: int,
        selected_clients: Sequence[int],
        state_before: Mapping[str, torch.Tensor],
        state_after: Mapping[str, torch.Tensor],
        update_hashes: Sequence[str],
    ) -> AuditRoundRecord:
        state_hash_before = self._state_hash(state_before)
        state_hash_after = self._state_hash(state_after)
        update_root = merkle_root(list(update_hashes))
        vrf_seed = json_digest(
            {
                "round_id": round_id,
                "state_hash_before": state_hash_before,
                "prev_chain_hash": self.chain_hash,
            }
        )
        record_payload = {
            "round_id": round_id,
            "selected_clients": list(map(int, selected_clients)),
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "update_root": update_root,
            "vrf_seed": vrf_seed,
            "prev_chain_hash": self.chain_hash,
        }
        next_chain_hash = json_digest(record_payload)
        record = AuditRoundRecord(chain_hash=next_chain_hash, **record_payload)
        self.rounds.append(record)
        self.chain_hash = next_chain_hash
        return record

    def log_repair_round(
        self,
        repair_id: int,
        selected_clients: Sequence[int],
        state_before: Mapping[str, torch.Tensor],
        state_after: Mapping[str, torch.Tensor],
        update_hashes: Sequence[str],
        target_basis: Optional[torch.Tensor] = None,
    ) -> AuditRepairRecord:
        state_hash_before = self._state_hash(state_before)
        state_hash_after = self._state_hash(state_after)
        update_root = merkle_root(list(update_hashes))
        target_basis_hash = tensor_digest(target_basis) if target_basis is not None else "0" * 64
        vrf_seed = json_digest(
            {
                "phase": "repair",
                "repair_id": repair_id,
                "state_hash_before": state_hash_before,
                "target_basis_hash": target_basis_hash,
                "prev_chain_hash": self.chain_hash,
            }
        )
        record_payload = {
            "repair_id": int(repair_id),
            "selected_clients": list(map(int, selected_clients)),
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "update_root": update_root,
            "vrf_seed": vrf_seed,
            "target_basis_hash": target_basis_hash,
            "prev_chain_hash": self.chain_hash,
        }
        next_chain_hash = json_digest(record_payload)
        record = AuditRepairRecord(chain_hash=next_chain_hash, **record_payload)
        self.repair_rounds.append(record)
        self.chain_hash = next_chain_hash
        return record

    def updates_for_client(self, client_id: int) -> List[ClientUpdateRecord]:
        return [r for r in self.client_updates if r.client_id == int(client_id)]

    def export(self, path: os.PathLike[str] | str, extra: Optional[Mapping[str, object]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "rounds": [asdict(r) for r in self.rounds],
            "repair_rounds": [asdict(r) for r in self.repair_rounds],
            "client_updates": [r.meta() for r in self.client_updates],
            "extra": dict(extra or {}),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def merkle_root(leaves: Sequence[str]) -> str:
    if not leaves:
        return sha256_bytes(b"")
    level = [leaf if len(leaf) == 64 else sha256_bytes(leaf.encode("utf-8")) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [sha256_bytes((level[i] + level[i + 1]).encode("utf-8")) for i in range(0, len(level), 2)]
    return level[0]


def deterministic_select_clients(
    client_ids: Sequence[int],
    count: int,
    seed_material: Mapping[str, object] | str,
    include_client: Optional[int] = None,
    exclude_clients: Optional[Sequence[int]] = None,
) -> List[int]:
    """Deterministically select clients from hash-ranked scores.

    This is a lightweight, auditable VRF-style scheduler for experiments: the
    seed is public and reproducible, while every selected client follows from a
    canonical hash ranking.
    """

    count = max(0, int(count))
    excluded = set(map(int, exclude_clients or []))
    include = None if include_client is None else int(include_client)
    if include is not None:
        excluded.discard(include)
    pool = [int(cid) for cid in client_ids if int(cid) not in excluded and int(cid) != include]
    seed = seed_material if isinstance(seed_material, str) else json_digest(seed_material)
    ranked = sorted(pool, key=lambda cid: json_digest({"seed": seed, "client_id": cid}))
    selected: List[int] = []
    if include is not None and include in [int(cid) for cid in client_ids]:
        selected.append(include)
    selected.extend(ranked[: max(0, count - len(selected))])
    return sorted(selected[:count])


def estimate_contribution(
    records: Sequence[ClientUpdateRecord],
    config: SRAuditConfig,
    template: Mapping[str, torch.Tensor],
    current_round: Optional[int] = None,
) -> TensorDict:
    if not records:
        return {name: torch.zeros_like(tensor) for name, tensor in template.items()}
    current_round = int(current_round if current_round is not None else max(r.round_id for r in records))
    contribution = {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in template.items()}
    for record in records:
        decay = config.decay ** max(0, current_round - record.round_id)
        scale = record.aggregation_weight * decay
        for name in contribution.keys():
            contribution[name].add_(record.shared_update[name].float(), alpha=scale)
    return contribution


def estimate_total_abs_contribution(
    records: Sequence[ClientUpdateRecord],
    template: Mapping[str, torch.Tensor],
    current_round: Optional[int],
    decay: float,
) -> TensorDict:
    total = {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in template.items()}
    if not records:
        return total
    current_round = int(current_round if current_round is not None else max(r.round_id for r in records))
    for record in records:
        scale = record.aggregation_weight * (decay ** max(0, current_round - record.round_id))
        for name in total.keys():
            total[name].add_(record.shared_update[name].float().abs(), alpha=abs(scale))
    return total


def make_mask(
    target_contribution: Mapping[str, torch.Tensor],
    config: SRAuditConfig,
    total_abs_contribution: Optional[Mapping[str, torch.Tensor]] = None,
    eps: float = 1.0e-12,
) -> TensorDict:
    mode = config.mask.lower()
    if mode == "full":
        return {name: torch.ones_like(tensor, dtype=torch.float32) for name, tensor in target_contribution.items()}

    if mode == "topk":
        flat_abs = flatten_tensors({name: t.abs() for name, t in target_contribution.items()})
        if flat_abs.numel() == 0:
            return {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in target_contribution.items()}
        keep_ratio = min(max(config.topk_ratio, 0.0), 1.0)
        kth = max(1, int(math.ceil(flat_abs.numel() * keep_ratio)))
        threshold = torch.topk(flat_abs, kth).values.min()
        return {name: (tensor.abs() >= threshold).float() for name, tensor in target_contribution.items()}

    if mode == "relative":
        if total_abs_contribution is None:
            raise ValueError("relative mask requires total_abs_contribution")
        mask: TensorDict = {}
        for name, tensor in target_contribution.items():
            dominance = tensor.abs() / (total_abs_contribution[name].abs() + eps)
            mask[name] = (dominance >= config.relative_threshold).float()
        return mask

    raise ValueError(f"Unsupported mask mode: {config.mask}")


def apply_mask(update: Mapping[str, torch.Tensor], mask: Mapping[str, torch.Tensor]) -> TensorDict:
    return {name: update[name].float() * mask[name].float() for name in update.keys()}


def mcr_remove_(
    model: torch.nn.Module,
    masked_contribution: Mapping[str, torch.Tensor],
    config: SRAuditConfig,
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> None:
    add_scaled_state_dict_(
        model,
        masked_contribution,
        scale=-float(config.mcr_strength),
        shared_keywords=shared_keywords,
        excluded_keywords=excluded_keywords,
    )


def build_target_subspace(
    records: Sequence[ClientUpdateRecord],
    template: Mapping[str, torch.Tensor],
    rank: int,
    mask: Optional[Mapping[str, torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """Return an orthonormal basis B with shape [p, q] for target update directions."""

    if rank <= 0 or not records:
        return None
    names = sorted(template.keys())
    masked_updates = [apply_mask(r.shared_update, mask) if mask is not None else r.shared_update for r in records]
    matrix = torch.stack([flatten_tensors(update, names) for update in masked_updates], dim=0)
    if matrix.numel() == 0:
        return None
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    max_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if max_rank <= 0:
        return None
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def project_orthogonal(vector: torch.Tensor, basis: Optional[torch.Tensor]) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return vector
    basis = basis.to(vector.device, vector.dtype)
    flat = vector.reshape(-1)
    return (flat - basis.matmul(basis.T.matmul(flat))).view_as(vector)


def unlearning_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """FedOSD UCE loss: bounded objective for active target-client forgetting."""

    probs = torch.softmax(logits, dim=1)
    true_probs = probs.gather(1, labels.view(-1, 1)).clamp(min=eps, max=1.0 - eps)
    return -torch.log(1.0 - true_probs / 2.0).mean()


def orthogonal_steepest_descent_direction(
    target_grad: torch.Tensor,
    retained_grads: Optional[torch.Tensor],
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Project the target UCE gradient onto the nullspace of retained gradients.

    This mirrors FedOSD's d = g_u - A^T(AA^T)^+ A g_u, then restores the
    original target-gradient norm.
    """

    gu = target_grad.detach().float().reshape(-1)
    if retained_grads is None or retained_grads.numel() == 0:
        return gu
    A = retained_grads.detach().float()
    if A.ndim == 1:
        A = A.unsqueeze(0)
    A = A.to(gu.device)
    gram = A.matmul(A.T)
    projected = gu - A.T.matmul(torch.linalg.pinv(gram).matmul(A.matmul(gu.unsqueeze(1)))).reshape(-1)
    norm = torch.linalg.norm(projected)
    gu_norm = torch.linalg.norm(gu)
    if norm <= eps or gu_norm <= eps:
        return gu
    return projected / norm * gu_norm


def project_against_direction(vector: torch.Tensor, direction: Optional[torch.Tensor], eps: float = 1.0e-12) -> torch.Tensor:
    """Remove the component of vector aligned with direction."""

    if direction is None or direction.numel() == 0:
        return vector
    flat = vector.reshape(-1).float()
    direction = direction.reshape(-1).to(flat.device, flat.dtype)
    denom = torch.dot(direction, direction)
    if denom <= eps:
        return vector
    projected = flat - torch.dot(flat, direction) / denom * direction
    return projected.view_as(vector)


def project_gradient_dict(
    grads: Mapping[str, torch.Tensor],
    basis: Optional[torch.Tensor],
    template: Mapping[str, torch.Tensor],
) -> TensorDict:
    if basis is None:
        return {name: tensor for name, tensor in grads.items()}
    names = sorted(template.keys())
    flat = flatten_tensors(grads, names).to(basis.device)
    projected = project_orthogonal(flat, basis)
    return unflatten_vector(projected.cpu(), {name: template[name].cpu() for name in names})


def apply_projected_gradients_(
    model: torch.nn.Module,
    basis: Optional[torch.Tensor],
    shared_keywords: Optional[Sequence[str]] = None,
    excluded_keywords: Optional[Sequence[str]] = None,
) -> None:
    """Project existing shared-parameter gradients away from the target subspace."""

    if basis is None:
        return
    params = named_shared_parameters(model, shared_keywords, excluded_keywords)
    grads = {name: _to_cpu_tensor(param.grad) for name, param in params if param.grad is not None}
    if not grads:
        return
    template = {name: _to_cpu_tensor(param) for name, param in params if param.grad is not None}
    projected = project_gradient_dict(grads, basis.cpu(), template)
    with torch.no_grad():
        for name, param in params:
            if param.grad is not None and name in projected:
                param.grad.copy_(projected[name].to(param.grad.device, param.grad.dtype))


def build_embedding_subspace(embeddings: torch.Tensor, rank: int) -> Optional[torch.Tensor]:
    """Return a low-dimensional embedding direction basis A with shape [d_z, k]."""

    if rank <= 0 or embeddings.numel() == 0:
        return None
    z = embeddings.detach().float()
    if z.ndim != 2:
        raise ValueError("embeddings must be rank-2 tensors [n, d]")
    z = z - z.mean(dim=0, keepdim=True)
    max_rank = min(rank, z.shape[0], z.shape[1])
    if max_rank <= 0:
        return None
    _, _, vh = torch.linalg.svd(z, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def direction_penalty(embeddings: torch.Tensor, direction_basis: Optional[torch.Tensor]) -> torch.Tensor:
    """Client-penalty repair term ||A^T z||^2 for public-calibration directions."""

    if direction_basis is None or direction_basis.numel() == 0:
        return embeddings.new_tensor(0.0)
    basis = direction_basis.to(embeddings.device, embeddings.dtype)
    z = embeddings - embeddings.mean(dim=0, keepdim=True)
    projected = z.matmul(basis)
    return projected.pow(2).mean()


def adversarial_confusion_loss(logits: torch.Tensor) -> torch.Tensor:
    """Encoder-side adversarial loss: push discriminator output toward 0.5."""

    target = torch.full_like(logits, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target)


def discriminator_loss(pre_logits: torch.Tensor, repair_logits: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(pre_logits)
    zeros = torch.zeros_like(repair_logits)
    return F.binary_cross_entropy_with_logits(pre_logits, ones) + F.binary_cross_entropy_with_logits(repair_logits, zeros)


def dv_mutual_information_bound(joint_scores: torch.Tensor, marginal_scores: torch.Tensor) -> torch.Tensor:
    """Donsker-Varadhan MI lower bound used as a minimization penalty."""

    return joint_scores.mean() - torch.logsumexp(marginal_scores, dim=0) + math.log(max(1, marginal_scores.numel()))


def proximal_loss(model: torch.nn.Module, reference_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    params = dict(model.named_parameters())
    for name, ref in reference_state.items():
        if name in params:
            losses.append(torch.sum((params[name] - ref.to(params[name].device, params[name].dtype)) ** 2))
    if not losses:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).sum()


def covariance_matrix(embeddings: torch.Tensor, lambda_white: float = 0.1) -> torch.Tensor:
    z = embeddings.detach().float()
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T.matmul(z) / max(1, z.shape[0] - 1)
    reg = lambda_white * torch.trace(cov) / max(1, cov.shape[0])
    return cov + reg * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)


def representation_audit_scores(
    target_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    lambda_white: float = 0.1,
) -> Dict[str, float]:
    """Black-box representation audit using variance, inner-product, and CKA scores."""

    target = target_embeddings.detach().float()
    reference = reference_embeddings.detach().float()
    if target.ndim != 2 or reference.ndim != 2:
        raise ValueError("embeddings must be rank-2 tensors [n, d]")
    cov_ref = covariance_matrix(reference, lambda_white)
    inv_cov = torch.linalg.pinv(cov_ref)
    diff = target.mean(dim=0) - reference.mean(dim=0)
    mean_mahalanobis = diff.unsqueeze(0).matmul(inv_cov).matmul(diff.unsqueeze(1)).item()

    target_centered = target - target.mean(dim=0, keepdim=True)
    ref_centered = reference - reference.mean(dim=0, keepdim=True)
    target_gram = target_centered.matmul(target_centered.T)
    ref_gram = ref_centered.matmul(ref_centered.T)
    min_n = min(target_gram.shape[0], ref_gram.shape[0])
    target_gram = target_gram[:min_n, :min_n]
    ref_gram = ref_gram[:min_n, :min_n]
    cka_num = (target_gram * ref_gram).sum()
    cka_den = torch.linalg.norm(target_gram) * torch.linalg.norm(ref_gram) + 1.0e-12

    pair_inner = target_centered.matmul(target_centered.T)
    off_diag = pair_inner[~torch.eye(pair_inner.shape[0], dtype=torch.bool, device=pair_inner.device)]
    return {
        "mean_mahalanobis": float(mean_mahalanobis),
        "coordinate_variance": float(target.var(dim=0).mean().item()),
        "pairwise_inner_abs_mean": float(off_diag.abs().mean().item()) if off_diag.numel() else 0.0,
        "linear_cka_to_reference": float((cka_num / cka_den).item()),
    }


def save_tensor_dict(path: os.PathLike[str] | str, tensors: Mapping[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: _to_cpu_tensor(tensor) for name, tensor in tensors.items()}, path)


def load_tensor_dict(path: os.PathLike[str] | str) -> TensorDict:
    return torch.load(path, map_location="cpu")
