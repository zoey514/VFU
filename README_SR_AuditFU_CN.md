# SR-AuditFU 中文说明

本项目实现的是面向共享表征式个性化联邦学习的可审计客户端遗忘方法。当前主方法已经按研究报告升级为 **SR-AuditFU-OSD**：在原 SR-AuditFU 的共享编码器审计、MCR、目标子空间投影 repair 基础上，融合 FedOSD 的 **Unlearning Cross-Entropy (UCE)**、**Orthogonal Steepest Descent (OSD)** 和 recovery 阶段梯度投影。

项目当前同时提供两条运行路径：

- standalone 路径：`system/experiments/standalone_cifar10_fedrep_srauditfu.py`，不依赖完整 PFLlib，适合在当前仓库直接跑 CIFAR-10 实验和 smoke test。
- PFLlib 路径：`system/flcore/servers/serverauditfu.py` 与 `system/flcore/clients/clientauditfu.py`，用于接入 PFLlib 的 server/client 框架。

## 方法概览

SR-AuditFU 的完整流程分为五个阶段。

1. 共享表征训练

   每个客户端模型由共享编码器和本地头组成：

   ```text
   f_i(x) = h_i(phi(x))
   ```

   其中 `phi` 是共享编码器，代码中对应 `model.base`；`h_i` 是第 `i` 个客户端的本地头，代码中对应 `model.head`。训练采用 FedRep 风格：先固定共享编码器训练本地头，再固定本地头训练共享编码器。服务器只聚合共享编码器的参数更新。

2. 训练日志与执行审计

   每轮训练会记录共享编码器更新、参与客户端、聚合前后模型哈希、更新 Merkle root、VRF-style seed 和 hash-chain。这样后续遗忘不是只给一个结果模型，而是能给出可复查的执行证据。

3. 目标客户端历史贡献移除 MCR

   当目标客户端 `target_client` 请求遗忘时，系统从日志中找出它参与过的所有轮次，按时间衰减累计它对共享编码器的贡献：

   ```text
   C_u = sum_t decay^(T-t) * aggregation_weight_t * delta_theta_u_t
   ```

   随后对贡献向量做 mask。目前支持三种 mask：

   - `full`：移除全部目标贡献。
   - `topk`：只移除绝对值最大的 top-k 坐标，当前默认配置。
   - `relative`：只移除目标贡献在总贡献中占比足够高的坐标。

   MCR 阶段执行：

   ```text
   theta_after_mcr = theta_before_unlearn - mcr_strength * masked_contribution
   ```

4. UCE + OSD 主动遗忘

   按 FedOSD 思路，目标客户端先使用 UCE 损失：

   ```text
   L_UCE = -log(1 - p_true / 2)
   ```

   相比直接梯度上升，UCE 有界，能降低梯度爆炸风险。然后计算目标客户端 UCE 梯度 `g_u`，并将其投影到 retained clients 梯度矩阵的正交补空间：

   ```text
   d = g_u - A^T (A A^T)^+ A g_u
   ```

   其中 `A` 是 retained clients 的共享编码器梯度矩阵。主方法在 MCR 后继续执行这个 UCE/OSD 更新。

5. 目标子空间约束 repair

   MCR 会降低目标客户端残留，但也可能伤害 retained clients 的效用。因此系统会在非目标客户端上做 repair。repair 不是普通 fine-tune，而是受到目标子空间约束：

   - 先用 masked target deltas 构造目标贡献子空间。
   - retained clients 本地执行带稳定项的 repair。
   - repair 聚合更新先剔除与 OSD 遗忘方向同向的分量，模拟 FedOSD recovery 投影，避免模型回退。
   - 服务器再将聚合向量正交投影到目标贡献子空间的补空间，避免 repair 把目标客户端方向重新加回来。

   standalone 路径中的 repair 损失为：

   ```text
   CE(retain labels)
   + lambda_kd * KL(pre logits || current logits)
   + lambda_feat * (embedding mean alignment + lambda_var * coordinate variance alignment)
   ```

6. 黑盒效果审计与综合评分

   遗忘后系统会输出多个维度指标：

   - retained utility：保留客户端准确率、macro-F1、新客户端头适配表现。
   - forgetting：task-inference AUC、MIA AUC、target CKA、参数距离、mask 稀疏度。
   - representation：坐标方差、成对内积、Mahalanobis 分数、目标是否落入 retained 分布区间。
   - execution audit：hash-chain、Merkle root、VRF seed、repair 记录是否可验证。
   - system cost：训练、repair、MCR、audit 时间，以及通信和存储估计。
   - report_metrics：按研究报告命名输出 `R-Acc`、`ASR`、`MIA-AUC`、`TIA-AUC`、`CKA`、`Dist-to-theta_T`、`AuditPass`。
   - audit_score：保留综合 gate 分数，用于判定实验是否可解释。

## 目录和文件职责

### 核心算法

`system/flcore/unlearning/auditfu.py`

负责 SR-AuditFU 的核心工具函数和数据结构：

- 区分共享编码器参数和本地头参数。
- 展平和恢复 tensor dict。
- 计算 shared update、模型哈希、tensor 哈希。
- 记录训练轮次 `AuditRoundRecord`。
- 记录 repair 轮次 `AuditRepairRecord`。
- 维护 hash-chain 和 Merkle root。
- 保存低秩日志 LR-Log。
- 计算目标客户端历史贡献。
- 生成 `full`、`topk`、`relative` mask。
- 执行 MCR。
- 计算 FedOSD UCE 损失。
- 计算 OSD 正交最速下降方向。
- 执行 recovery 阶段方向投影。
- 从 masked target deltas 构造目标贡献子空间。
- 将 repair update 投影到目标子空间的正交补空间。
- 计算表征审计分数。

### PFLlib Server

`system/flcore/servers/serverauditfu.py`

负责 PFLlib 版本的服务端流程：

- 正常联邦训练。
- 每轮记录共享编码器更新。
- 收到遗忘请求后定位目标客户端历史记录。
- 执行 MCR。
- 构造 masked target subspace。
- 在 retained clients 上执行 repair。
- 聚合并投影 repair update。
- 导出 evidence JSON。

### PFLlib Client

`system/flcore/clients/clientauditfu.py`

负责 PFLlib 版本客户端侧训练和 repair：

- 复用 FedRep 风格本地训练。
- 在 repair 阶段保留 pre-unlearn encoder 作为参考。
- 添加 feature mean 和 coordinate variance 稳定项。
- 支持 adversarial confusion、DV mutual information、proximal loss、direction penalty 等增强项。

### Standalone 多数据集实验

`system/experiments/standalone_cifar10_fedrep_srauditfu.py`

这是当前仓库最容易直接运行的入口，功能包括：

- 下载或读取 MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100、FEMNIST 代理数据。
- Dirichlet 或 pathological non-IID 客户端划分。
- small CNN encoder 或 ResNet-18 encoder，默认使用 GroupNorm，避免 BatchNorm running statistics 在联邦聚合中失效。
- FedRep 风格训练。
- 确定性可审计客户端选择。
- MCR + masked target subspace。
- UCE + OSD 主动遗忘。
- retained repair + FedOSD recovery 投影 + target subspace server projection。
- 多维度指标计算。
- 输出 `metrics.json`、`metrics_flat.csv`、`evidence.json`。

### PFLlib 启动器

`system/experiments/run_srauditfu.py`

负责把实验参数转发给 PFLlib 的 `system/main.py`。如果使用完整 PFLlib 框架，通常从这个文件或 shell 脚本启动。

### 运行脚本

`system/scripts/run_srauditfu_cifar10.sh`

提供旧版 CIFAR-10 + Dirichlet non-IID 的示例命令。

`system/scripts/run_srauditfu_osd_cifar10_pat.sh`

提供按研究报告和 FedOSD README 对齐的 CIFAR-10 pathological `NC=2` 示例命令。

`system/scripts/sweep_resnet18_cifar10_lrs.sh`

提供 ResNet-18 CIFAR-10 学习率 sweep 示例。

## 环境要求

项目主要依赖：

- Python 3.10 或更高版本。
- PyTorch。
- torchvision。
- NumPy。

如果使用 standalone CIFAR-10 路径，当前仓库已经可以直接运行，不需要完整 PFLlib checkout。

如果第一次运行本地没有 CIFAR-10 数据，需要加 `--download`。在当前受限环境中，建议优先使用已经存在的 `data/` 目录，避免网络下载失败。

## 快速验证

最小 smoke test：

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

这个命令只用于确认代码链路能跑通，不用于报告正式结果。它会执行 1 轮训练和 1 轮 repair，并输出：

```text
results/smoke_report_mods/evidence.json
results/smoke_report_mods/metrics.json
results/smoke_report_mods/metrics_flat.csv
```

成功运行后，重点检查：

- `execution_audit.exec_audit_pass` 是否为 `true`。
- `execution_audit.round_records` 是否等于训练轮数。
- `execution_audit.repair_round_records` 是否等于 repair 轮数。
- `baselines` 是否包含 `Retrain`、`NoUnlearn`、`Fine-tune`、`MCR-only`、`MCR+Repair`、`FedOSD-Adapted`、`SR-AuditFU`、`SR-AuditFU-OSD` 中已启用的方法。
- `report_metrics` 是否包含 `R-Acc`、`ASR`、`MIA-AUC`、`TIA-AUC`、`CKA`、`Dist-to-theta_T`、`AuditPass`。
- `audit_score.score_valid` 是否为 `true`。如果为 `false`，先看 `audit_score.invalid_reasons`，不要直接解释遗忘指标。

## 研究报告版实验设计

新报告要求实验设置与 FedOSD 尽量对齐，同时保留共享表征个性化 FL 和可审计设计。当前代码的默认设置已经调整为：

- 数据集：`mnist`、`fashionmnist`、`cifar10`、`cifar100`、`femnist`。
- 划分：`pathological` 和 `dirichlet`。
- pathological 默认 `classes_per_client=2`，对应 FedOSD README 的 `NC=2`。
- 客户端数默认 `N=100`。
- 参与率默认 `C=0.1`，即每轮约 10 个客户端参与。
- 预遗忘训练轮数默认 `global_rounds=50`，对应报告中“100 轮训练、50 轮后触发遗忘”的前半段。
- repair 默认 `repair_rounds=20`。
- mask 默认 top-10%，即 `auditfu_topk_ratio=0.1`。
- 贡献子空间 PCA/SVD rank 默认 `auditfu_subspace_rank=20`。
- OSD 学习率默认 `osd_lr=0.0004`，参考 FedOSD README 的 unlearning 命令。

注意：`--dataset femnist` 当前使用 torchvision `EMNIST(split="byclass")` 作为可运行代理，并继续使用本脚本的 Dirichlet/pathological 联邦划分；它不是 LEAF FEMNIST 的原始 writer-natural split。

推荐的 CIFAR-10 pathological `NC=2` 命令：

```bash
bash system/scripts/run_srauditfu_osd_cifar10_pat.sh
```

等价展开命令：

```bash
PYTHONPATH=system python -u system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --dataset cifar10 \
  --model small_cnn \
  --device auto \
  --num_clients 100 \
  --join_ratio 0.1 \
  --global_rounds 50 \
  --total_rounds 100 \
  --repair_rounds 20 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 200 \
  --lr_head 0.05 \
  --lr_encoder 0.05 \
  --split_mode pathological \
  --classes_per_client 2 \
  --target_client 0 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.1 \
  --auditfu_subspace_rank 20 \
  --osd_lr 0.0004 \
  --osd_retain_clients 10 \
  --repair_strength 0.2 \
  --repair_kd_lambda 0.5 \
  --repair_feat_lambda 0.5 \
  --repair_var_lambda 0.1 \
  --auditfu_log_dir results/srauditfu_osd_cifar10_pat_n100_nc2_c01
```

如果机器没有 GPU，这个命令会很慢。调试时建议加：

```bash
--num_clients 10 --global_rounds 2 --repair_rounds 1 --baseline_rounds 1 --skip_retrain_baseline --max_train_samples 1000 --max_test_samples 500
```

报告要求的 baseline 当前对应如下：

- `Retrain`：从头训练 retained clients。
- `NoUnlearn`：不执行遗忘。
- `Fine-tune`：UCE-only target update，不做 OSD 和 repair。
- `MCR-only`：只做贡献移除。
- `MCR+Repair`：MCR 后做 KD/feature repair，但不做 OSD 和目标子空间投影。
- `FedOSD-Adapted`：直接对共享编码器做 UCE/OSD，不做 MCR 和 SR-AuditFU 子空间 repair。
- `SR-AuditFU`：原始 MCR + target-subspace projected repair，不含 UCE/OSD。
- `SR-AuditFU-OSD`：本文方法，MCR + UCE/OSD + recovery 方向投影 + target-subspace repair。

## ResNet-18 版本

如果希望使用更强的共享编码器，可以选择 ResNet-18。根据当前 10 客户端、全量 CIFAR-10、50 轮训练、10 轮 repair 的结果，主方法的 utility 和 audit 已经通过，但 `TIA-AUC`、retained `CKA` 和 target/retain CKA ratio 仍需要加强。因此推荐优先使用下面的 tuned 配置：

```bash
PYTHONPATH=system python -u system/experiments/standalone_cifar10_fedrep_srauditfu.py \
  --dataset cifar10 \
  --model resnet18 \
  --device cuda \
  --num_clients 10 \
  --join_ratio 0.2 \
  --participation_mode force_target \
  --target_client 0 \
  --global_rounds 50 \
  --repair_rounds 10 \
  --repair_early_stop_patience 6 \
  --head_epochs 1 \
  --encoder_epochs 1 \
  --batch_size 32 \
  --embedding_dim 128 \
  --split_mode pathological \
  --classes_per_client 2 \
  --auditfu_mask topk \
  --auditfu_topk_ratio 0.2 \
  --auditfu_mcr_strength 1.5 \
  --auditfu_subspace_rank 20 \
  --osd_lr 0.0004 \
  --osd_max_batches 1 \
  --osd_retain_clients 10 \
  --repair_strength 0.15 \
  --repair_kd_lambda 0.5 \
  --repair_kd_temp 2.0 \
  --repair_feat_lambda 1.0 \
  --repair_var_lambda 0.2 \
  --repair_prox_lambda 0.02 \
  --repair_subspace_lambda 1.5 \
  --retrain_rounds 100 \
  --auditfu_log_dir results/resnet18_cifar10_pat_forgetting_tuned
```

也可以直接运行脚本：

```bash
bash system/scripts/run_resnet18_cifar10_pat_forgetting_tuned.sh
```

这组参数相对上一轮结果做了四个调整：

- `auditfu_mcr_strength=1.5` 和 `repair_subspace_lambda=1.5`：增强目标客户端贡献移除和目标子空间约束，优先改善 `TIA-AUC` 与 `Target-CKA`。
- `repair_strength=0.15`：减小每轮 repair 写回步长，降低 retained 表征被过度扰动的风险。
- `repair_feat_lambda=1.0`、`repair_var_lambda=0.2`、`repair_prox_lambda=0.02`：增强 retained 表征稳定约束，目标是提升 retained `CKA`。
- `retrain_rounds=100`：避免 `Retrain` baseline 因训练不足而明显偏低。

注意：ResNet-18 在 CPU 上会明显更慢，建议有 GPU 时再跑完整配置。

## 输出文件说明

每次 standalone 运行会在 `--auditfu_log_dir` 下写出三个文件。

### evidence.json

这是审计证据包，包含：

- `config`：本次 SR-AuditFU 配置。
- `rounds`：训练阶段每轮审计记录。
- `repair_rounds`：repair 阶段每轮审计记录。
- `client_updates`：客户端共享编码器更新的摘要记录。
- `extra`：数据集、算法、目标客户端、basis rank、指标文件名等元数据。

重点字段：

- `state_hash_before`：本轮聚合前共享编码器哈希。
- `state_hash_after`：本轮聚合后共享编码器哈希。
- `update_root`：本轮客户端 update hash 的 Merkle root。
- `vrf_seed`：可重放调度使用的公开 seed。
- `prev_chain_hash`：上一条记录的 chain hash。
- `chain_hash`：当前记录的 chain hash。
- `target_basis_hash`：repair 轮次绑定的目标子空间哈希。

### metrics.json

这是嵌套结构的完整指标文件，适合程序读取和论文表格整理。

主要分组：

- `utility`
- `forgetting`
- `representation`
- `execution_audit`
- `report_metrics`
- `baselines`
- `audit_score`
- `system_cost`
- `threshold_checks`
- `metadata`

### metrics_flat.csv

这是扁平化指标文件，一行一个指标，适合导入 Excel、pandas 或画图脚本。

## 关键指标解释

### utility

用于衡量 retained clients 的可用性。

- `weighted_acc`：按样本数加权的总体准确率。
- `mean_client_acc`：客户端平均准确率。
- `retain_mean_acc`：非目标客户端平均准确率。
- `target_client_acc`：目标客户端准确率。
- `macro_f1`：宏平均 F1。
- `new_client_adaptation_*`：新客户端只训练本地头后的适配能力。

### forgetting

用于衡量目标客户端信息是否被削弱。

- `task_inference_auc_post`：遗忘后的 task inference AUC，越接近 0.5 越好。
- `mia_auc_post_loss`：基于 loss 的 membership inference AUC，越接近 0.5 越好。
- `target_cka_pre_to_post`：目标客户端遗忘前后 embedding 相似度，越低表示变化越大。
- `retain_cka_pre_to_post_mean`：retained clients 遗忘前后 embedding 相似度，越高表示保留越稳定。
- `target_to_retain_cka_ratio`：目标变化与 retained 变化的相对比值。
- `mask_sparsity`：mask 稀疏度。

### representation

用于黑盒表征审计。

- `coordinate_variance`：目标 embedding 的坐标方差。
- `pairwise_inner_abs_mean`：目标 embedding 成对内积绝对值均值。
- `mean_mahalanobis`：目标 embedding 与参考 retained embedding 的 Mahalanobis 距离。
- `linear_cka_to_reference`：目标 embedding 与参考 retained embedding 的 CKA。
- `target_pairwise_inner_within_retain_95ci`：目标统计量是否落入 retained 分布的 95% 区间。

### execution_audit

用于确认执行过程是否可验证。

- `exec_audit_pass`：训练和 repair hash-chain 是否通过验证。
- `round_records`：训练轮记录数。
- `repair_round_records`：repair 轮记录数。
- `client_update_records`：客户端 update 记录数。
- `chain_verification_rate`：chain 验证通过率。
- `merkle_root_presence_rate`：训练轮 Merkle root 存在率。
- `vrf_seed_presence_rate`：训练轮 VRF seed 存在率。
- `repair_vrf_seed_presence_rate`：repair 轮 VRF seed 存在率。

### report_metrics

这是最贴近研究报告表格的指标分组。

- `R-Acc`：retained clients 平均准确率，越高越好。
- `ASR`：目标客户端攻击成功率，这里用目标客户端测试准确率近似，越低表示遗忘越彻底。
- `MIA-AUC`：成员推断 AUC，越接近 0.5 越好。
- `TIA-AUC`：任务推断 AUC，越接近 0.5 越好。
- `CKA`：retained clients 遗忘前后表征相似度，越高表示保留越稳定。
- `Dist-to-theta_T`：遗忘模型和遗忘前污染模型的相对参数距离，越大表示离原目标影响越远。
- `AuditPass`：执行审计是否通过，通过为 1，否则为 0。

`report_metrics.baselines` 会对每个 baseline 输出同一组核心对比项：

- `R-Acc`、`ASR`：utility 和目标客户端残留表现。
- `MIA-AUC`、`TIA-AUC`：遗忘侧攻击指标，越接近 0.5 越好。
- `CKA`：retained clients 表征保持程度，越高越好。
- `Target-CKA`：目标客户端遗忘前后表征相似度，越低通常表示目标影响移除更彻底。
- `Target/Retain-CKA`：目标变化与 retained 稳定性的相对关系，越低越符合“忘 target、保 retained”的目标。
- `Dist-to-theta_T`：相对于遗忘前共享模型的参数距离。

这项改动用于避免只按 retained accuracy 排序。比如 `FedOSD-Adapted` 可能拥有更高的 `R-Acc`，但如果 `TIA-AUC`、`Target-CKA` 或 `Target/Retain-CKA` 更差，就不能说明它整体优于 SR-AuditFU-OSD。

### baselines

该分组保存所有启用方法的原始指标、history、诊断指标和说明。正式表格建议优先读取 `report_metrics.baselines`，需要细节时再查看 `baselines.<method>.diagnostics`。

### audit_score

根据报告中的综合评分思想输出：

```text
score_sr_auditfu =
  exec_gate
  * retain_score
  * (0.4 * forgetting_score_tia
     + 0.3 * forgetting_score_mia
     + 0.3 * forgetting_score_cka)
```

其中：

- `exec_gate`：执行审计是否通过，通过为 1，否则为 0。
- `retain_score`：retained utility 保留比例。
- `forgetting_score_tia`：task inference AUC 越接近 0.5 分数越高。
- `forgetting_score_mia`：MIA AUC 越接近 0.5 分数越高。
- `forgetting_score_cka`：目标 CKA 越低分数越高。

这个分数适合做实验主表中的整体参考，但不能替代分项指标。论文分析时应同时报告 retained utility、MIA、task inference、CKA 和系统成本。

## PFLlib 集成方式

如果要把本方法放入完整 PFLlib，需要在 PFLlib 的 `system/main.py` 中加入：

```python
from flcore.servers.serverauditfu import SRAuditFU

if args.algorithm == "SRAuditFU":
    server = SRAuditFU(args, i)
```

如果 PFLlib 的 argparse 不接受未知参数，还需要加入：

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

然后可以使用：

```bash
bash system/scripts/run_srauditfu_cifar10.sh
```

## 常见问题

### 为什么只审计共享编码器，不审计本地头？

在 FedRep/FedPer 这类个性化联邦学习中，本地头属于客户端私有参数。客户端退出后，本地头可以直接在客户端侧删除。真正被多个客户端共同训练、也可能携带目标客户端残留的是共享编码器。因此本项目把遗忘对象集中在 `model.base`。

### 为什么默认用 top-k mask？

full mask 会移除所有目标贡献，可能对 retained utility 伤害更大。top-k mask 只处理贡献绝对值最大的坐标，更符合“只移除最可能携带目标客户端信息的共享表示方向”的设计。当前默认 `topk_ratio=0.2`，即使用 20% 坐标。

### repair 为什么要投影？

如果只在 retained clients 上普通 fine-tune，模型可能沿着目标客户端历史贡献方向回弹。目标子空间投影会把聚合 repair update 中落在目标贡献子空间内的分量去掉，降低“遗忘后又恢复目标残留”的风险。

### smoke test 指标不好是否代表方法失败？

不代表。smoke test 使用极少样本和极少轮数，只用于验证代码链路。正式分析需要使用完整数据、足够训练轮数、多 seed 和基线对比。

### CPU 能跑吗？

能跑，但全量 CIFAR-10、50 轮训练、10 轮 repair 在 CPU 上会比较慢。ResNet-18 版本尤其建议使用 GPU。

## 当前实现边界

当前代码是研究原型，已经支持轻量可审计执行链和黑盒效果审计，但还不是强密码学系统：

- VRF 当前是可重放的 hash-based deterministic scheduler，不是完整 RFC 9381 ECVRF。
- Merkle root 和 hash-chain 用于研究级 evidence package，不包含完整链上或第三方公证流程。
- low-rank log 目前用于存储估计和摘要，不是完整压缩恢复系统。
- few-shot proxy mixup warm-up 目前没有作为默认路径启用，主流程是 MCR + masked subspace + retained repair。

这些边界不影响当前实验运行，但在论文或报告中需要明确说明。
