# 项目经历（基于 Issue #29139 / PR #35209）

## 项目名称
vLLM TP-MoE 通信优化项目（torch.compile Pass）

## 项目角色
项目经理 / 技术项目负责人（开源协作场景）

## 项目背景
在 vLLM 的 TP（Tensor Parallel）+ MoE 场景下，原有执行路径存在冗余通信（`all_reduce -> rms_norm -> chunk`），影响性能与扩展效率。Issue #29139 提出通过 `torch.compile` 自定义 Pass 重写图结构，以降低不必要通信并兼容现有并行策略。

## 目标与范围
- 推动实现并落地 `SequenceParallelismMoEPass`，以 `reduce_scatter` 替换关键路径中的冗余通信。
- 新增独立开关 `enable_sp_moe`，支持与常规 SP 解耦配置和灰度启用。
- 保证与 AsyncTP、EP/TP/DP 组合及残差路径兼容。
- 完成单测、服务压测和任务评测闭环验证。

## 我的关键工作（项目经理视角）
- **需求澄清与方案收敛**：基于 Issue 讨论，明确“优化通信 + 保持正确性 + 支持动态 shape”的目标边界，推动从“图内 padding”转向“运行时 padding”等可落地路径。
- **任务拆解与协同推进**：围绕 Pass 设计、并行配置、运行时逻辑、测试覆盖拆分子任务，协调贡献者与 reviewer（含跨团队）迭代。
- **风险识别与外部依赖管理**：对动态 shape + padding 引发的编译问题组织最小复现与问题上报（PyTorch issue），并同步回流到实现策略。
- **评审闭环与质量门禁**：跟进审查意见（包含残差形态、PP 传输 gating、平台开关保护等），推动问题逐项关闭。
- **验证与结果呈现**：组织 UT、`vllm serve`、`lm_eval`、e2e benchmark 的验证流程，确保 PR 描述包含可复现实验命令与结果。

## 量化产出（来自 PR #35209）
> 注：以下统计为 2026-03-17 抓取 PR 页面时的数据快照，后续如有 rebase 或追加提交可能变化。
- 代码变更规模：**12 个文件**，**20 次提交**，新增约 **994 行**。
- 单测：`tests/compile/passes/distributed/test_sequence_parallelism_moe.py` 共 **20/20 通过**。
- 端到端验证：完成服务侧压测与 `gsm8k` 任务评估，形成可复用的性能对比与回归基线。

## 项目价值
- 在 TP-MoE 路径上引入面向编译优化的通信重写能力，降低冗余通信风险。
- 建立“问题提出 -> 方案实现 -> 跨团队评审 -> 指标验证”的开源项目推进闭环。
- 为后续 SP / SP-MoE 融合优化、残差通信消除等演进方向提供工程基础。

## 简历可用精炼版（可直接粘贴）
- 主导 vLLM 在 TP+MoE 场景下的通信优化项目（Issue #29139 / PR #35209），推动引入 `SequenceParallelismMoEPass` 与独立配置开关 `enable_sp_moe`，实现从方案设计到评审落地的全流程管理。
- 协调多方贡献者与 reviewer，完成动态 shape、padding、残差与并行传输等关键风险收敛，建立跨仓协作闭环（含 PyTorch 最小复现问题上报）。
- 组织 UT/服务压测/lm_eval/e2e 多层验证并沉淀可复现实验命令，保障变更可验证、可回归、可迭代。
