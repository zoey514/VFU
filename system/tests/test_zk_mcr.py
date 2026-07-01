"""Lightweight self-checks for the ZK-MCR prototype verifier.

Run directly without pytest:

    PYTHONPATH=system python -B system/tests/test_zk_mcr.py
"""

from __future__ import annotations

import copy

import torch

from flcore.unlearning.auditfu import ClientUpdateRecord, SRAuditConfig, apply_mask, estimate_contribution, tensor_digest
from flcore.unlearning.zk_mcr import ZKMCRPrototypeVerifier


def _fixture(with_records: bool = True):
    template = {"w": torch.zeros(3, dtype=torch.float32)}
    config = SRAuditConfig(target_client=0, decay=0.5, mcr_strength=2.0, mask="topk", topk_ratio=1.0)
    records = [
        ClientUpdateRecord(
            round_id=1,
            client_id=0,
            aggregation_weight=0.5,
            shared_update={"w": torch.tensor([1.0, -2.0, 3.0])},
            update_hash=tensor_digest(torch.tensor([1.0, -2.0, 3.0])),
        ),
        ClientUpdateRecord(
            round_id=2,
            client_id=0,
            aggregation_weight=0.25,
            shared_update={"w": torch.tensor([2.0, 0.0, -1.0])},
            update_hash=tensor_digest(torch.tensor([2.0, 0.0, -1.0])),
        ),
    ]
    target_records = records if with_records else None
    target_contribution = estimate_contribution(records, config, template, current_round=2)
    mask = {"w": torch.ones(3, dtype=torch.float32)}
    masked_contribution = apply_mask(target_contribution, mask)
    theta_before = {"w": torch.tensor([10.0, 20.0, 30.0])}
    theta_after = {"w": theta_before["w"] - config.mcr_strength * masked_contribution["w"]}
    return config, template, records, target_records, target_contribution, mask, masked_contribution, theta_before, theta_after


def _prove_and_verify(**overrides):
    (
        config,
        template,
        records,
        target_records,
        target_contribution,
        mask,
        masked_contribution,
        theta_before,
        theta_after,
    ) = _fixture(with_records=overrides.pop("with_records", True))
    values = {
        "target_contribution": target_contribution,
        "mask": mask,
        "masked_contribution": masked_contribution,
        "theta_before": theta_before,
        "theta_after": theta_after,
        "target_records": target_records,
        "config": config,
        "template": template,
        "current_round": 2,
    }
    values.update(overrides)
    verifier = ZKMCRPrototypeVerifier()
    proof = verifier.prove(
        theta_before=values["theta_before"],
        target_contribution=values["target_contribution"],
        mask=values["mask"],
        masked_contribution=values["masked_contribution"],
        theta_after=values["theta_after"],
        mcr_strength=config.mcr_strength,
        target_records=values["target_records"],
        config=values["config"],
        template=values["template"],
        current_round=values["current_round"],
    )
    return verifier.verify(
        proof=proof,
        theta_before=values["theta_before"],
        target_contribution=values["target_contribution"],
        mask=values["mask"],
        masked_contribution=values["masked_contribution"],
        theta_after=values["theta_after"],
        mcr_strength=config.mcr_strength,
        target_records=values["target_records"],
        config=values["config"],
        template=values["template"],
        current_round=values["current_round"],
    )


def test_output_and_mask_consistency_pass():
    result = _prove_and_verify()
    assert result["verified"] is True
    assert result["levels"]["output_consistency"]["verified"] is True
    assert result["levels"]["mask_consistency"]["verified"] is True


def test_mask_consistency_fails_when_masked_contribution_is_tampered():
    (
        _config,
        _template,
        _records,
        _target_records,
        _target_contribution,
        _mask,
        masked_contribution,
        _theta_before,
        _theta_after,
    ) = _fixture()
    tampered = copy.deepcopy(masked_contribution)
    tampered["w"] = tampered["w"] + torch.tensor([0.1, 0.0, 0.0])
    result = _prove_and_verify(masked_contribution=tampered)
    assert result["levels"]["mask_consistency"]["verified"] is False


def test_output_consistency_fails_when_theta_after_is_tampered():
    *_, theta_after = _fixture()
    tampered = copy.deepcopy(theta_after)
    tampered["w"] = tampered["w"] + torch.tensor([0.1, 0.0, 0.0])
    result = _prove_and_verify(theta_after=tampered)
    assert result["levels"]["output_consistency"]["verified"] is False


def test_contribution_consistency_pass_with_target_records():
    result = _prove_and_verify()
    assert result["verification_scope"] == "prototype_level_1_2_3"
    assert result["levels"]["contribution_consistency"]["available"] is True
    assert result["levels"]["contribution_consistency"]["verified"] is True


def test_contribution_consistency_unavailable_without_target_records():
    result = _prove_and_verify(with_records=False, target_records=None, config=None, template=None)
    assert result["verified"] is True
    assert result["verification_scope"] == "prototype_level_1_2"
    assert result["partial_verification"] is True
    assert result["levels"]["contribution_consistency"]["available"] is False


def main() -> None:
    tests = [
        test_output_and_mask_consistency_pass,
        test_mask_consistency_fails_when_masked_contribution_is_tampered,
        test_output_consistency_fails_when_theta_after_is_tampered,
        test_contribution_consistency_pass_with_target_records,
        test_contribution_consistency_unavailable_without_target_records,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
