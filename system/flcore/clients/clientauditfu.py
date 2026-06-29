"""PFLlib client wrapper for SR-AuditFU.

This client extends FedRep-style local training with optional representation
repair losses.  It keeps the PFLlib convention that the shared encoder lives in
``model.base`` and the personalized head lives in ``model.head``; custom models
still work as long as their parameter names can be separated by the keyword
rules in ``flcore.unlearning.auditfu``.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.unlearning.auditfu import (
    SRAuditConfig,
    adversarial_confusion_loss,
    direction_penalty,
    dv_mutual_information_bound,
    proximal_loss,
)

try:
    from flcore.clients.clientrep import clientRep as _BaseClient
except Exception:  # pragma: no cover - lets this file compile outside PFLlib.
    _BaseClient = object


class FeatureDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        hidden_dim = max(16, min(hidden_dim, feature_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.detach() if not features.requires_grad else features)


class MINet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        hidden_dim = max(16, min(hidden_dim, feature_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z1, z2], dim=1)).squeeze(1)


class clientAuditFU(_BaseClient):
    """FedRep-compatible client with repair-mode hooks."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.auditfu_config = SRAuditConfig.from_args(args)
        self.pre_encoder: Optional[nn.Module] = None
        self.discriminator: Optional[FeatureDiscriminator] = None
        self.mi_net: Optional[MINet] = None
        self.direction_basis: Optional[torch.Tensor] = None
        self.repair_mode = False

    def enable_repair(self, pre_encoder_state=None):
        self.repair_mode = True
        if hasattr(self.model, "base"):
            self.pre_encoder = copy.deepcopy(self.model.base)
            if pre_encoder_state is not None:
                self.pre_encoder.load_state_dict(pre_encoder_state, strict=False)
            self.pre_encoder.to(self.device)
            self.pre_encoder.eval()

    def set_direction_basis(self, direction_basis: Optional[torch.Tensor]):
        self.direction_basis = None if direction_basis is None else direction_basis.detach().float().cpu()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "base"):
            features = self.model.base(x)
            return torch.flatten(features, 1) if features.ndim > 2 else features
        if hasattr(self.model, "encoder"):
            features = self.model.encoder(x)
            return torch.flatten(features, 1) if features.ndim > 2 else features
        raise AttributeError("SR-AuditFU client requires model.base or model.encoder for feature repair.")

    def _forward_head(self, features: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "head"):
            return self.model.head(features)
        return self.model(features)

    def _ensure_repair_modules(self, feature_dim: int):
        if self.discriminator is None:
            self.discriminator = FeatureDiscriminator(feature_dim).to(self.device)
        if self.mi_net is None:
            self.mi_net = MINet(feature_dim).to(self.device)

    def train_repair(self, reference_state=None):
        """Repair shared encoder on retained clients with Adv + MI + prox terms."""

        trainloader = self.load_train_data()
        self.model.train()
        if self.pre_encoder is None:
            self.enable_repair()

        max_local_epochs = self.local_epochs
        start_time = time.time()

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                features = self._extract_features(x)
                self._ensure_repair_modules(features.shape[1])
                logits = self._forward_head(features)
                loss = self.loss(logits, y)

                with torch.no_grad():
                    pre_features = self.pre_encoder(x)
                    pre_features = torch.flatten(pre_features, 1) if pre_features.ndim > 2 else pre_features

                if self.auditfu_config.lambda_feat > 0.0:
                    mean_loss = F.mse_loss(features.mean(dim=0), pre_features.mean(dim=0))
                    var_loss = F.mse_loss(
                        features.var(dim=0, unbiased=False),
                        pre_features.var(dim=0, unbiased=False),
                    )
                    loss = loss + self.auditfu_config.lambda_feat * (
                        mean_loss + self.auditfu_config.lambda_var * var_loss
                    )

                disc_logits = self.discriminator(features)
                loss = loss + self.auditfu_config.lambda_adv * adversarial_confusion_loss(disc_logits)

                if self.mi_net is not None:
                    rolled = pre_features.roll(shifts=1, dims=0)
                    joint_scores = self.mi_net(features, pre_features)
                    marginal_scores = self.mi_net(features, rolled)
                    loss = loss + self.auditfu_config.lambda_mi * dv_mutual_information_bound(
                        joint_scores, marginal_scores
                    )

                if reference_state is not None:
                    loss = loss + self.auditfu_config.lambda_prox * proximal_loss(self.model, reference_state)

                if self.direction_basis is not None:
                    loss = loss + self.auditfu_config.lambda_dir * direction_penalty(features, self.direction_basis)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
