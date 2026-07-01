"""PFLlib server implementation for SR-AuditFU experiments.

The server is designed to be dropped into PFLlib as ``flcore/servers/serverauditfu.py``
and selected from ``main.py`` with an ``SRAuditFU`` algorithm branch.  It reuses
FedRep/FedPer-style clients and adds:

* per-round shared encoder update logging with hash-chain records;
* MCR contribution removal for a target client;
* SVD target subspace construction;
* server-side target-subspace orthogonal projection during retained-client repair;
* auditable evidence export.

This PFLlib path follows the same main method as the standalone runner:
SR-AuditFU = auditable logging + sparse MCR + target-subspace orthogonal
retained repair. The projection can be disabled for the SR-AuditFU-Core
ablation with ``--disable_target_subspace_projection``.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from flcore.clients.clientauditfu import clientAuditFU
from flcore.unlearning.auditfu import (
    AuditLogger,
    SRAuditConfig,
    apply_mask,
    build_target_subspace,
    estimate_contribution,
    estimate_total_abs_contribution,
    flatten_tensors,
    make_mask,
    mcr_remove_,
    project_orthogonal,
    save_tensor_dict,
    shared_state_dict,
    subtract_state_dict,
    unflatten_vector,
)

try:
    from flcore.servers.serverrep import FedRep as _BaseFedRep
except Exception:  # pragma: no cover - lets this file compile outside PFLlib.
    try:
        from flcore.servers.serverbase import Server as _BaseFedRep
    except Exception:
        _BaseFedRep = object


class SRAuditFU(_BaseFedRep):
    """Shared-Representation Auditable Federated Unlearning server."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.auditfu_config = SRAuditConfig.from_args(args)
        self.audit_logger = AuditLogger(self.auditfu_config)
        self.target_basis: Optional[torch.Tensor] = None
        self.shared_template: Optional[Dict[str, torch.Tensor]] = None
        self.pre_unlearn_state: Optional[Dict[str, torch.Tensor]] = None
        self.direction_basis: Optional[torch.Tensor] = self._load_direction_basis()
        self.repair_rounds = int(getattr(args, "auditfu_repair_rounds", max(1, getattr(args, "global_rounds", 10) // 10)))
        self.target_client = int(getattr(args, "target_client", self.auditfu_config.target_client))
        self.enable_target_subspace_projection = bool(getattr(args, "enable_target_subspace_projection", True))

        if hasattr(self, "set_slow_clients"):
            self.set_slow_clients()
        if hasattr(self, "set_clients"):
            self.set_clients(clientAuditFU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating SR-AuditFU server and clients.")

    def _load_direction_basis(self) -> Optional[torch.Tensor]:
        path = self.auditfu_config.direction_basis_path
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"auditfu_direction_basis_path not found: {path}")
        basis = torch.load(path, map_location="cpu")
        if isinstance(basis, dict):
            basis = basis.get("basis", basis.get("direction_basis"))
        if not torch.is_tensor(basis):
            raise TypeError("direction basis file must contain a tensor or {'basis': tensor}")
        return basis.detach().float().cpu()

    def _client_id(self, client) -> int:
        return int(getattr(client, "id", getattr(client, "client_id", -1)))

    def _shared_state(self) -> Dict[str, torch.Tensor]:
        return shared_state_dict(self.global_model)

    def _client_shared_update(self, before: Mapping[str, torch.Tensor], client) -> Dict[str, torch.Tensor]:
        after = shared_state_dict(client.model)
        return subtract_state_dict(after, before)

    def _aggregation_weight_for(self, client) -> float:
        train_samples = float(getattr(client, "train_samples", 1.0))
        total = sum(float(getattr(c, "train_samples", 1.0)) for c in self.selected_clients)
        return train_samples / max(total, 1.0)

    def _aggregate_shared_updates(self, updates: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]):
        aggregate = {name: torch.zeros_like(tensor) for name, tensor in self.shared_template.items()}
        for update, weight in zip(updates, weights):
            for name in aggregate.keys():
                aggregate[name].add_(update[name].float(), alpha=float(weight))
        return aggregate

    def _apply_server_projection(self, aggregate_update: Mapping[str, torch.Tensor]):
        if not self.enable_target_subspace_projection:
            return aggregate_update
        if self.target_basis is None or self.shared_template is None:
            return aggregate_update
        names = sorted(self.shared_template.keys())
        flat_update = flatten_tensors(aggregate_update, names)
        projected = project_orthogonal(flat_update, self.target_basis.cpu())
        return unflatten_vector(projected, {name: self.shared_template[name].cpu() for name in names})

    def _apply_shared_update_to_global(self, update: Mapping[str, torch.Tensor]):
        params = dict(self.global_model.named_parameters())
        with torch.no_grad():
            for name, delta in update.items():
                if name in params:
                    params[name].add_(delta.to(params[name].device, params[name].dtype))

    def _send_repair_state(self):
        if not hasattr(self, "clients"):
            return
        pre_encoder_state = None
        if hasattr(self.global_model, "base"):
            pre_encoder_state = copy.deepcopy(self.global_model.base.state_dict())
        for client in self.clients:
            if hasattr(client, "enable_repair"):
                client.enable_repair(pre_encoder_state)
            if hasattr(client, "set_direction_basis"):
                client.set_direction_basis(self.direction_basis)

    def train(self):
        self.shared_template = self._shared_state()
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            state_before = self._shared_state()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            client_updates = []
            weights = []
            update_hashes = []
            selected_ids = []
            for client in self.selected_clients:
                client.train()
                update = self._client_shared_update(state_before, client)
                weight = self._aggregation_weight_for(client)
                record = self.audit_logger.log_client_update(i, self._client_id(client), weight, update)
                client_updates.append(update)
                weights.append(weight)
                update_hashes.append(record.update_hash)
                selected_ids.append(self._client_id(client))

            aggregate = self._aggregate_shared_updates(client_updates, weights)
            self._apply_shared_update_to_global(aggregate)
            state_after = self._shared_state()
            self.audit_logger.log_round(i, selected_ids, state_before, state_after, update_hashes)

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / max(1, len(self.Budget[1:])))

        self.save_results()
        self.save_global_model()
        self.unlearn(self.target_client)

    def unlearn(self, target_client: int):
        """Run SR-AuditFU after normal shared-representation training."""

        self.auditfu_config.target_client = int(target_client)
        self.shared_template = self._shared_state()
        self.pre_unlearn_state = copy.deepcopy(self.global_model.state_dict())
        target_records = self.audit_logger.updates_for_client(target_client)
        current_round = max([r.round_id for r in self.audit_logger.client_updates], default=0)

        target_contribution = estimate_contribution(
            target_records,
            self.auditfu_config,
            self.shared_template,
            current_round=current_round,
        )
        total_abs = estimate_total_abs_contribution(
            self.audit_logger.client_updates,
            self.shared_template,
            current_round=current_round,
            decay=self.auditfu_config.decay,
        )
        mask = make_mask(target_contribution, self.auditfu_config, total_abs)
        masked_contribution = apply_mask(target_contribution, mask)

        save_dir = Path(self.auditfu_config.log_dir)
        save_tensor_dict(save_dir / f"target_{target_client}_contribution.pt", target_contribution)
        save_tensor_dict(save_dir / f"target_{target_client}_mask.pt", mask)

        mcr_remove_(self.global_model, masked_contribution, self.auditfu_config)
        self.target_basis = build_target_subspace(
            target_records,
            self.shared_template,
            self.auditfu_config.subspace_rank,
            mask=mask,
        )
        self._send_repair_state()
        self.repair_retained_clients(target_client)

        evidence_path = save_dir / f"target_{target_client}_evidence.json"
        self.audit_logger.export(
            evidence_path,
            extra={
                "target_client": int(target_client),
                "target_round_count": len(target_records),
                "mask": self.auditfu_config.mask,
                "subspace_rank": 0 if self.target_basis is None else int(self.target_basis.shape[1]),
                "target_subspace_projection_enabled": bool(self.enable_target_subspace_projection),
                "repair_rounds": self.repair_rounds,
            },
        )
        print(f"SR-AuditFU evidence exported to {evidence_path}")

    def repair_retained_clients(self, target_client: int):
        retained_clients = [c for c in self.clients if self._client_id(c) != int(target_client)]
        reference_state = copy.deepcopy(self.global_model.state_dict())

        for r in range(self.repair_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.selected_clients = [c for c in self.selected_clients if self._client_id(c) != int(target_client)]
            if not self.selected_clients:
                self.selected_clients = retained_clients[: max(1, int(self.join_clients))]

            state_before = self._shared_state()
            self.send_models()
            client_updates = []
            weights = []
            update_hashes = []
            selected_ids = []
            for client in self.selected_clients:
                if hasattr(client, "train_repair"):
                    client.train_repair(reference_state=reference_state)
                else:
                    client.train()
                update = self._client_shared_update(state_before, client)
                weight = self._aggregation_weight_for(client)
                record = self.audit_logger.log_client_update(
                    self.global_rounds + r + 1,
                    self._client_id(client),
                    weight,
                    update,
                )
                client_updates.append(update)
                weights.append(weight)
                update_hashes.append(record.update_hash)
                selected_ids.append(self._client_id(client))

            aggregate = self._aggregate_shared_updates(client_updates, weights)
            aggregate = self._apply_server_projection(aggregate)
            self._apply_shared_update_to_global(aggregate)
            state_after = self._shared_state()
            self.audit_logger.log_repair_round(r, selected_ids, state_before, state_after, update_hashes, self.target_basis)
            self.Budget.append(time.time() - s_t)
            print(f"SR-AuditFU repair round {r + 1}/{self.repair_rounds}, time cost {self.Budget[-1]:.2f}s")

        if hasattr(self, "evaluate"):
            print("\nEvaluate unlearned model")
            self.evaluate()
