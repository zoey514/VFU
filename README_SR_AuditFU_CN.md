# SR-AuditFU 中文说明

## 1. 方法定位

SR-AuditFU 是一个面向共享表征式个性化联邦学习的轻量可审计联邦遗忘框架。当前论文主线不是默认的零知识证明系统，而是：

```text
SR-AuditFU = Auditable Contribution Logging
           + Sparse Target Contribution Removal
           + Target-Subspace Orthogonal Repair
           + retained-client KD / feature / prox repair
```

其中默认主方法是 `SR-AuditFU`。它在目标客户端贡献移除后，继续在 retained clients 上修复 utility，并把 repair update 投影到目标贡献子空间的正交补空间，降低 repair 阶段重新恢复目标客户端影响的风险。

`SR-AuditFU-Core` 保留为 no-projection ablation：它有可审计日志、稀疏 MCR 和 retained-client repair，但没有 target-subspace orthogonal projection。

本项目默认审计层由 hash-chain、Merkle root、deterministic scheduler 和 `evidence.json` 组成。这是轻量执行审计证据，不等同于零知识证明。ZK-MCR 只是可选增强模块，用于证明 MCR 算术执行关系，不证明完整训练或完整 repair。

## 2. 适用场景

SR-AuditFU 适用于共享表征式个性化联邦学习，例如 FedRep、FedPer 或类似“共享 encoder + 个性化 head”的设置。

- 服务器聚合共享表征参数，例如 `model.base`、`encoder`、`backbone`。
- 客户端保留本地个性化 head，例如 `model.head`、`classifier`。
- 遗忘对象是目标客户端对共享表征的历史贡献。
- retained clients 的目标是尽量保留 utility 和表征结构。

当前 standalone 实验入口是：

```text
system/experiments/standalone_cifar10_fedrep_srauditfu.py
```

PFLlib 接入路径包括：

```text
system/flcore/unlearning/auditfu.py
system/flcore/servers/serverauditfu.py
system/flcore/clients/clientauditfu.py
```

## 3. 核心模块

### 3.1 Auditable Contribution Logging

该模块记录每轮共享表征更新，并生成可复核证据：

- `ClientUpdateRecord`：记录客户端 id、聚合权重、共享更新 hash、可选低秩日志。
- `AuditRoundRecord`：记录训练轮次、选中客户端、模型状态 hash、Merkle root、VRF-style seed、hash-chain。
- `AuditRepairRecord`：记录 repair 轮次，并把 `target_basis_hash` 绑定到 repair 证据中。
- deterministic scheduler：用公开 seed 和 hash 排序复现客户端选择。
- `evidence.json`：导出训练记录、repair 记录、客户端更新 metadata 和实验 extra 信息。

注意：hash-chain 和 Merkle root 只能证明导出的证据包内部一致，不能自动等价于第三方公证或零知识证明。如果服务器没有把链头提交给第三方或可信时间戳系统，它仍可能事后伪造一整套自洽日志。

### 3.2 Sparse Target Contribution Removal

Sparse Target Contribution Removal 使用目标客户端历史共享表征更新估计其贡献：

```text
target_contribution = sum_t aggregation_weight_t * decay^(T-t) * delta_theta_u_t
```

然后生成 mask：

- `full`：移除全部估计贡献。
- `topk`：按贡献绝对值保留 top-k 坐标，默认适合控制遗忘扰动范围。
- `relative`：按目标贡献相对总贡献的 dominance 选择坐标。

最后执行 masked contribution removal：

```text
masked_contribution = mask * target_contribution
theta_after_mcr = theta_before_unlearn - mcr_strength * masked_contribution
```

### 3.3 Target-Subspace Orthogonal Repair

MCR 可能降低 retained clients 的 utility，因此 SR-AuditFU 会在 retained clients 上执行 repair。默认 repair 包括：

- CE：保留 retained clients 的分类能力。
- KD：让修复后模型贴近遗忘前教师模型 logits。
- feature mean / variance stability：稳定 retained 表征分布。
- prox：限制 repair 后参数偏离过大。

SR-AuditFU 的关键点是：repair update 聚合后，服务器会把它投影到目标贡献子空间的正交补空间：

```text
repair_update_projected = repair_update - B (B^T repair_update)
```

其中 `B` 是由目标客户端历史贡献构造的 target subspace。这样做的动机是：普通 retained-client fine-tune 虽然能恢复 utility，但可能沿目标客户端历史贡献方向“回弹”，重新恢复应被遗忘的影响。Target-subspace orthogonal repair 是默认主方法的一部分，不再只是 baseline。

### 3.4 Optional ZK-MCR Enhancement

`system/flcore/unlearning/zk_mcr.py` 提供 ZK-ready proof specification 和 deterministic prototype verifier。

它的证明关系只覆盖 MCR 算术正确性：

```text
target_contribution = sum_t aggregation_weight_t * decay^(T-t) * delta_theta_u_t
masked_contribution = mask * target_contribution
theta_after_mcr = theta_before_unlearn - mcr_strength * masked_contribution
hash(theta_after_mcr) == public output commitment
```

当前默认实现不是 production zkSNARK。除非接入真实外部 ZK backend，否则它只是 prototype verifier 和 proof metadata 生成器。

ZK-MCR 明确不证明：

- local training correctness；
- full federated aggregation correctness；
- retained-client repair correctness；
- 完整联邦训练过程；
- 完整遗忘流程。

## 4. 方法变体和 Baselines

| 方法 | 类型 | 默认运行 | 说明 |
| --- | --- | --- | --- |
| `NoUnlearn` | baseline | 是 | 不做遗忘，保留污染模型。 |
| `MCR-only` | baseline | 是 | 只执行 sparse target contribution removal，不做 repair。 |
| `SR-AuditFU` | 主方法 | 是 | auditable logging + sparse MCR + target-subspace orthogonal repair + KD/feature/prox repair。 |
| `SR-AuditFU-Core` | ablation | 非 skip baseline 时运行 | 无 projection 的轻量消融：auditable logging + sparse MCR + KD/feature/prox repair。 |
| `Fine-tune` | baseline | 非 skip baseline 时运行 | FedOSD-style UCE-only target update，不做 OSD 和 retained repair。 |
| `FedOSD-Adapted` | 外部 baseline | 非 skip baseline 时运行 | 将 FedOSD 的 UCE/OSD 思路适配到共享 encoder。 |
| `Retrain` | oracle baseline | 未设置 `--skip_retrain_baseline` 时运行 | 去掉目标客户端后从头训练或按设定预算训练，用作 oracle 参考。 |
| `MCR+Repair` | redundant ablation | 默认不运行 | 与 `SR-AuditFU-Core` 路径高度重叠，仅在 `--include_redundant_baselines` 时运行。 |

如果显式加入：

```bash
--disable_target_subspace_projection
```

则主路径会退化为 no-projection variant，语义上等价于 `SR-AuditFU-Core`。

## 5. 指标解释

### Utility

- `R-Acc`：retained clients 平均准确率，越高越好。
- `Target-Acc`：target client accuracy。它不是 attack success rate，不能叫 ASR，也不能直接代表攻击成功率。
- `ASR_deprecated_alias_of_Target_Acc`：兼容旧脚本的 deprecated alias，不建议在论文中使用。

### Forgetting

- `MIA-AUC`：成员推断 AUC，越接近 0.5 越接近不可区分。
- `TIA-AUC`：任务/客户端推断 AUC，越接近 0.5 越接近不可区分。当前判定建议使用 `abs(TIA-AUC - 0.5) <= task_auc_tolerance`，避免 AUC 过低被误读。
- `Target-CKA`：目标客户端遗忘前后表征相似度，较低通常表示目标表征影响变化更明显。
- `Retain-CKA` 或 `CKA`：retained clients 遗忘前后表征相似度，越高表示 retained 表征越稳定。
- `Target/Retain-CKA`：目标变化和 retained 稳定性的相对指标，越低越符合“忘 target、保 retained”。

### Dist-to-theta_T

`Dist-to-theta_T` 只衡量当前共享 encoder 参数相对污染模型 `theta_T` 的偏离。距离更大不必然代表遗忘更好。它是辅助诊断指标，必须和 `MIA-AUC`、`TIA-AUC`、`Target-CKA`、`Retain-CKA` 以及 retained utility 一起解释。

### Retrain Consistency

当 `Retrain` baseline 可用时，SR-AuditFU 会额外输出：

- `Agreement-to-Retrain`：在 retained clients 测试集上，SR-AuditFU 与 Retrain 预测类别一致率。
- `KL-to-Retrain`：Retrain logits 到 SR-AuditFU logits 的平均 KL divergence，越低越接近。
- `CKA-to-Retrain`：二者 shared encoder embeddings 的 CKA。
- `ParamDist-to-Retrain`：二者共享 encoder 参数距离。

如果跳过 Retrain，这些字段会输出 `null` 或 `not_available`，不会导致程序崩溃。

### AuditPass 和 audit_score

- `AuditPass`：hash-chain、Merkle root、repair 记录等执行审计是否通过。
- `audit_score.score_sr_auditfu`：综合 summary score，用于快速筛查实验质量。

`audit_score` 不能替代分项指标。论文主表和分析必须同时报告 retained utility、MIA、TIA、CKA、Retrain consistency 和系统成本。

当 Retrain 可用时，summary score 会纳入 retrain consistency：

```text
0.25 * TIA score
+ 0.25 * MIA score
+ 0.20 * Target-CKA score
+ 0.15 * Retain-CKA score
+ 0.15 * Retrain consistency score
```

当 Retrain 不可用时，summary score 自动回退到不含 consistency 的版本。

## 6. 快速运行

### 6.1 Smoke Test

```bash
PYTHONPATH=system python -B system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --device cpu \
  --num_clients 3 \
  --join_ratio 0.67 \
  --global_rounds 1 \
  --repair_rounds 1 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 64 \
  --embedding_dim 32 \
  --max_train_samples 300 \
  --max_test_samples 150 \
  --max_audit_batches 2 \
  --new_client_adapt_steps 1 \
  --alpha 0.5 \
  --baseline_rounds 1 \
  --retrain_rounds 1 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.2 \
  --auditfu_log_dir results/smoke_report_mods
```

该命令用于确认代码链路能跑通，不用于正式论文结果。主方法默认启用 target-subspace projection。

输出文件：

```text
results/smoke_report_mods/evidence.json
results/smoke_report_mods/metrics.json
results/smoke_report_mods/metrics_flat.csv
results/smoke_report_mods/checkpoints/
```

### 6.2 ZK-MCR Prototype Smoke Test

ZK-MCR 默认关闭。开启 prototype verifier：

```bash
PYTHONPATH=system python -B system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --device cpu \
  --num_clients 3 \
  --join_ratio 0.67 \
  --global_rounds 1 \
  --repair_rounds 1 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 64 \
  --embedding_dim 32 \
  --max_train_samples 300 \
  --max_test_samples 150 \
  --max_audit_batches 2 \
  --new_client_adapt_steps 1 \
  --alpha 0.5 \
  --baseline_rounds 1 \
  --retrain_rounds 1 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.2 \
  --enable_zk_mcr \
  --zk_mcr_mode prototype \
  --auditfu_log_dir results/smoke_zk_mcr
```

开启后，`metrics.json` 和 `evidence.json` 会包含：

```json
"zk_mcr": {
  "enabled": true,
  "mode": "prototype",
  "proved_relation": "MCR execution correctness only",
  "proves_training": false,
  "proves_repair": false,
  "proves_mcr": true
}
```

如果未开启，则输出：

```json
"zk_mcr": {
  "enabled": false,
  "reason": "ZK-MCR is an optional enhancement; the default audit layer is hash-chain/Merkle evidence."
}
```

### 6.3 ResNet-18 全量 CIFAR-10 推荐脚本

```bash
bash system/scripts/run_resnet18_cifar10_pat_forgetting_tuned.sh
```

脚本默认：

- 10 clients；
- pathological 2 classes/client；
- full CIFAR-10；
- `join_ratio=0.4`；
- `global_rounds=100`，`global_min_rounds=50`；
- `repair_rounds=100`；
- early stop 为连续 5 轮准确率下降才停止；
- top-k 20% mask；
- target-subspace projection 默认启用；
- ZK-MCR 默认关闭。

## 7. 当前实现边界

当前实现需要按以下边界解释：

- 默认审计是 hash-chain / Merkle evidence，不是零知识证明。
- ZK-MCR 是可选增强，只证明 MCR arithmetic correctness。
- ZK-MCR 不证明 local training correctness。
- ZK-MCR 不证明 full federated aggregation correctness。
- ZK-MCR 不证明 retained-client repair correctness。
- 当前 `vrf_seed` 是 VRF-style deterministic seed，不是完整 RFC 9381 ECVRF 实现。
- 如果没有第三方公证、可信时间戳或外部提交机制，Merkle/hash-chain 不能防止服务器事后伪造一整套自洽日志。
- `Dist-to-theta_T` 只是参数偏离诊断，不能单独作为遗忘成功证据。
- `Target-Acc` 不是 ASR，不应在论文中写成 attack success rate。

## 8. 文件职责

- `system/experiments/standalone_cifar10_fedrep_srauditfu.py`：standalone CIFAR/FedRep 实验入口，负责训练、MCR、SR-AuditFU repair、baselines、指标和 evidence 输出。
- `system/flcore/unlearning/auditfu.py`：审计日志、贡献估计、mask、MCR、target subspace、projection 和表示审计工具。
- `system/flcore/unlearning/zk_mcr.py`：ZK-MCR proof relation spec 和 prototype verifier，不是 production zkSNARK。
- `system/flcore/servers/serverauditfu.py`：PFLlib server 接入，负责共享 encoder 聚合、审计记录、MCR 和 retained repair projection。
- `system/flcore/clients/clientauditfu.py`：PFLlib client 接入，负责 retained repair 的本地训练辅助损失。
- `system/scripts/*.sh`：常用实验脚本。
