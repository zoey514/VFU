"""ZK-ready proof specification and prototype verifier for MCR correctness.

This module defines a ZK-ready proof relation and deterministic prototype
verifier for MCR correctness.

Prototype mode verifies:
- Level 1: output consistency
- Level 2: mask consistency
- optionally Level 3: time-decayed contribution consistency if target records
  are provided

It is not a production zkSNARK implementation and does not provide
zero-knowledge security unless a real backend is plugged in. It does not prove
local training correctness, full federated aggregation correctness,
retained-client repair correctness, or full unlearning correctness.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch

from flcore.unlearning.auditfu import (
    ClientUpdateRecord,
    SRAuditConfig,
    TensorDict,
    apply_mask,
    estimate_contribution,
    json_digest,
    subtract_state_dict,
    tensor_digest,
)


def _state_commitment(state: Mapping[str, torch.Tensor]) -> str:
    return json_digest({name: tensor_digest(tensor) for name, tensor in sorted(state.items())})


@dataclass(frozen=True)
class ZKMCRSpec:
    """Public proof relation for MCR arithmetic correctness only."""

    relation: str = (
        "target_contribution = sum_t aggregation_weight_t * decay^(T-t) * delta_theta_u_t; "
        "masked_contribution = mask * target_contribution; "
        "theta_after_mcr = theta_before_unlearn - mcr_strength * masked_contribution; "
        "hash(theta_after_mcr) == public output commitment"
    )
    proves_mcr: bool = True
    proves_training: bool = False
    proves_repair: bool = False
    proves_full_federated_aggregation: bool = False


@dataclass(frozen=True)
class ZKMCRProofSpec:
    """Deterministic prototype proof object for the MCR relation."""

    mode: str
    proved_relation: str
    relation_levels: Dict[str, bool]
    input_commitments: Dict[str, str | float]
    output_commitment: str
    quantization: Dict[str, int | str]
    notes: Dict[str, str]
    proof_hash: str
    proof_size_bytes: int
    proof_time_seconds: float


@dataclass(frozen=True)
class ZKMCRVerificationResult:
    """Structured result for deterministic prototype verification."""

    verified: bool
    mode: str
    verification_scope: str
    proved_relation: str
    levels: Dict[str, Dict[str, object]]
    max_abs_error: float
    output_commitment_matches: bool
    verification_time_seconds: float


class ZKMCRPrototypeVerifier:
    """Verify MCR arithmetic deterministically without claiming zkSNARK security."""

    def __init__(self, quant_scale: int = 1_000_000):
        self.quant_scale = int(quant_scale)
        if self.quant_scale <= 0:
            raise ValueError("zk_mcr_quant_scale must be positive")
        self.spec = ZKMCRSpec()

    def _quantized_digest(self, tensors: Mapping[str, torch.Tensor]) -> str:
        payload = {}
        for name, tensor in sorted(tensors.items()):
            quantized = torch.round(tensor.detach().float().cpu() * self.quant_scale).to(torch.int64)
            payload[name] = tensor_digest(quantized.float())
        return json_digest(payload)

    def _max_abs_error(
        self,
        left: Mapping[str, torch.Tensor],
        right: Mapping[str, torch.Tensor],
    ) -> float:
        max_abs_error = 0.0
        for name in left.keys():
            if name not in right:
                return float("inf")
            error = (left[name].detach().float().cpu() - right[name].detach().float().cpu()).abs().max().item()
            max_abs_error = max(max_abs_error, float(error))
        for name in right.keys():
            if name not in left:
                return float("inf")
        return float(max_abs_error)

    def _contribution_inputs_available(
        self,
        target_records: Optional[Sequence[ClientUpdateRecord]],
        config: Optional[SRAuditConfig],
        template: Optional[Mapping[str, torch.Tensor]],
    ) -> bool:
        return target_records is not None and config is not None and template is not None

    def prove(
        self,
        theta_before: Mapping[str, torch.Tensor],
        target_contribution: Mapping[str, torch.Tensor],
        mask: Mapping[str, torch.Tensor],
        masked_contribution: Mapping[str, torch.Tensor],
        theta_after: Mapping[str, torch.Tensor],
        mcr_strength: float,
        target_records: Optional[Sequence[ClientUpdateRecord]] = None,
        config: Optional[SRAuditConfig] = None,
        template: Optional[Mapping[str, torch.Tensor]] = None,
        current_round: Optional[int] = None,
    ) -> ZKMCRProofSpec:
        """Create deterministic metadata for the MCR proof relation."""

        start = time.time()
        contribution_available = self._contribution_inputs_available(target_records, config, template)
        relation_levels = {
            "output_consistency": True,
            "mask_consistency": True,
            "contribution_consistency": bool(contribution_available),
            "mask_generation_correctness": False,
            "merkle_inclusion": False,
            "production_zksnark": False,
        }
        notes = {}
        if not contribution_available:
            notes["contribution_consistency"] = "target_records/config/template not provided"
        notes["mask_generation_correctness"] = (
            "prototype mode does not prove top-k/relative mask generation correctness"
        )
        notes["merkle_inclusion"] = "prototype mode does not prove Merkle inclusion of target updates"
        notes["production_zksnark"] = "no production zkSNARK backend is attached"
        input_commitments: Dict[str, str | float] = {
            "theta_before_unlearn": _state_commitment(theta_before),
            "target_contribution": self._quantized_digest(target_contribution),
            "mask": self._quantized_digest(mask),
            "masked_contribution": self._quantized_digest(masked_contribution),
            "mcr_strength": float(mcr_strength),
            "target_record_count": float(len(target_records)) if target_records is not None else 0.0,
        }
        if current_round is not None:
            input_commitments["current_round"] = float(current_round)
        output_commitment = _state_commitment(theta_after)
        proof_hash = json_digest(
            {
                "spec": self.spec.relation,
                "relation_levels": relation_levels,
                "input_commitments": input_commitments,
                "output_commitment": output_commitment,
                "quant_scale": self.quant_scale,
                "notes": notes,
            }
        )
        proof_size_bytes = len(proof_hash.encode("utf-8")) + sum(
            len(str(key).encode("utf-8")) + len(str(value).encode("utf-8"))
            for key, value in input_commitments.items()
        )
        return ZKMCRProofSpec(
            mode="prototype",
            proved_relation="MCR execution correctness prototype",
            relation_levels=relation_levels,
            input_commitments=input_commitments,
            output_commitment=output_commitment,
            quantization={"scheme": "round_float_to_int", "scale": self.quant_scale},
            notes=notes,
            proof_hash=proof_hash,
            proof_size_bytes=int(proof_size_bytes),
            proof_time_seconds=float(time.time() - start),
        )

    def verify(
        self,
        proof: ZKMCRProofSpec,
        theta_before: Mapping[str, torch.Tensor],
        target_contribution: Mapping[str, torch.Tensor],
        mask: Mapping[str, torch.Tensor],
        masked_contribution: Mapping[str, torch.Tensor],
        theta_after: Mapping[str, torch.Tensor],
        mcr_strength: float,
        target_records: Optional[Sequence[ClientUpdateRecord]] = None,
        config: Optional[SRAuditConfig] = None,
        template: Optional[Mapping[str, torch.Tensor]] = None,
        current_round: Optional[int] = None,
        atol: float = 1.0e-5,
    ) -> Dict[str, object]:
        """Run Level 1/2 and optional Level 3 deterministic prototype checks."""

        start = time.time()
        expected_delta: TensorDict = {}
        for name, tensor in masked_contribution.items():
            expected_delta[name] = -float(mcr_strength) * tensor.detach().float().cpu()
        observed_delta = subtract_state_dict(theta_after, theta_before)
        output_error = self._max_abs_error(observed_delta, expected_delta)
        output_matches = _state_commitment(theta_after) == proof.output_commitment
        output_verified = bool(output_error <= atol and output_matches)

        expected_masked = apply_mask(target_contribution, mask)
        mask_error = self._max_abs_error(masked_contribution, expected_masked)
        mask_verified = bool(mask_error <= atol)

        levels: Dict[str, Dict[str, object]] = {
            "output_consistency": {
                "available": True,
                "verified": output_verified,
                "max_abs_error": float(output_error),
                "atol": float(atol),
            },
            "mask_consistency": {
                "available": True,
                "verified": mask_verified,
                "max_abs_error": float(mask_error),
                "atol": float(atol),
            },
        }

        contribution_available = self._contribution_inputs_available(target_records, config, template)
        contribution_verified: Optional[bool] = None
        contribution_error: Optional[float] = None
        if contribution_available:
            estimated = estimate_contribution(target_records or [], config, template, current_round)
            contribution_error = self._max_abs_error(target_contribution, estimated)
            contribution_verified = bool(contribution_error <= atol)
            levels["contribution_consistency"] = {
                "available": True,
                "verified": contribution_verified,
                "target_record_count": int(len(target_records or [])),
                "max_abs_error": float(contribution_error),
                "atol": float(atol),
            }
        else:
            levels["contribution_consistency"] = {
                "available": False,
                "verified": None,
                "reason": "target_records/config/template not provided",
            }

        levels["mask_generation_correctness"] = {
            "available": False,
            "verified": None,
            "reason": "prototype mode does not prove top-k/relative mask generation correctness",
        }
        levels["merkle_inclusion"] = {
            "available": False,
            "verified": None,
            "reason": "prototype mode does not prove Merkle inclusion of target updates",
        }
        levels["production_zksnark"] = {
            "available": False,
            "verified": None,
            "reason": "no production zkSNARK backend is attached",
        }

        verified = bool(output_verified and mask_verified and (contribution_verified is not False))
        verification_scope = "prototype_level_1_2_3" if contribution_available else "prototype_level_1_2"
        result = ZKMCRVerificationResult(
            verified=verified,
            mode=proof.mode,
            verification_scope=verification_scope,
            proved_relation=proof.proved_relation,
            levels=levels,
            max_abs_error=float(max(output_error, mask_error, contribution_error or 0.0)),
            output_commitment_matches=bool(output_matches),
            verification_time_seconds=float(time.time() - start),
        )
        payload = asdict(result)
        payload.update(
            {
                "proves_mcr": True,
                "proves_mcr_output_consistency": True,
                "proves_mask_consistency": True,
                "proves_contribution_consistency": bool(contribution_available),
                "proves_mask_generation_correctness": False,
                "proves_merkle_inclusion": False,
                "proves_training": False,
                "proves_repair": False,
                "production_zksnark": False,
                "partial_verification": bool(not contribution_available),
            }
        )
        return payload


def disabled_zk_mcr_metadata() -> Dict[str, object]:
    """Return metadata for the default no-ZK setting."""

    return {
        "enabled": False,
        "reason": "ZK-MCR is optional; default audit layer is hash-chain/Merkle evidence.",
        "proves_mcr": False,
        "proves_mcr_output_consistency": False,
        "proves_mask_consistency": False,
        "proves_contribution_consistency": False,
        "proves_mask_generation_correctness": False,
        "proves_merkle_inclusion": False,
        "proves_training": False,
        "proves_repair": False,
        "production_zksnark": False,
    }
