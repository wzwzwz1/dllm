# MDLM Entropy-Priority 实现拆分清单

## 1. 目标

本清单用于把当前 `MDLM` 优化方案拆成可落地的工程任务，并满足一个硬性要求：

> 每一处新增优化/模块都必须可以通过参数独立开关，方便做消融实验。

实现目标不是一次性上完整系统，而是分阶段引入：

- `tentative / delayed finalize`
- `targeted remask`
- `structure-aware priority`
- `early-sparse quota`
- `repair stage`

并保证每个部分都可单独关闭。

---

## 2. 涉及文件

核心文件建议先只改这几个：

- [dllm/core/samplers/mdlm.py](/Users/wz/code/dllm/dllm/core/samplers/mdlm.py)
- [dllm/core/samplers/utils.py](/Users/wz/code/dllm/dllm/core/samplers/utils.py)
- [examples/llada/sample.py](/Users/wz/code/dllm/examples/llada/sample.py)
- [dllm/pipelines/llada/eval.py](/Users/wz/code/dllm/dllm/pipelines/llada/eval.py)

如果需要更干净的实现，第二阶段再考虑新增一个辅助文件，例如：

- `/Users/wz/code/dllm/dllm/core/samplers/tentative.py`

第一版不强制拆文件，优先保证改动小、可控。

---

## 3. 参数化原则

所有新增能力都遵守下面的参数设计原则：

- 默认值必须保持当前 baseline 行为不变
- 每个模块有独立的 `enable_*` 参数
- 行为阈值、预算、阶段比例也要单独参数化
- 日志统计单独开关，避免默认污染性能

推荐命名风格：

- `enable_tentative_commit`
- `enable_targeted_remask`
- `enable_structure_priority`
- `enable_priority_age_bonus`
- `enable_early_sparse_quota`
- `enable_repair_stage`
- `enable_entropy_priority`
- `enable_sampler_diagnostics`

这样后续做消融时可以清楚写成：

```text
baseline
+ tentative
+ tentative + remask
+ tentative + remask + structure
+ tentative + remask + structure + sparse_quota
```

---

## 4. 配置层改造清单

在 [mdlm.py](/Users/wz/code/dllm/dllm/core/samplers/mdlm.py) 的 `MDLMSamplerConfig` 中新增以下参数。

### 4.1 总开关

- `enable_entropy_priority: bool = False`
- `enable_tentative_commit: bool = False`
- `enable_targeted_remask: bool = False`
- `enable_structure_priority: bool = False`
- `enable_priority_age_bonus: bool = False`
- `enable_early_sparse_quota: bool = False`
- `enable_repair_stage: bool = False`
- `enable_sampler_diagnostics: bool = False`

### 4.2 现有 entropy-priority 参数保留

- `entropy_min_tokens_per_step: int = 0`
- `entropy_early_ratio: float = 0.3`
- `entropy_top_k: int | None = None`

要求：

- 当 `enable_entropy_priority=False` 时，这些参数即使被传入，也不应改变 baseline 行为

### 4.3 tentative / finalize 参数

- `tentative_budget_ratio: float = 0.1`
- `tentative_min_hold_steps: int = 1`
- `tentative_stable_steps: int = 2`
- `tentative_max_hold_steps: int = 3`
- `tentative_final_prob_thresh: float = 0.82`
- `tentative_final_margin_thresh: float = 0.35`

### 4.4 remask 参数

- `remask_rollback_prob_thresh: float = 0.45`
- `remask_flip_thresh: int = 2`
- `remask_only_tentative: bool = True`
- `repair_final_budget_ratio: float = 0.0`
- `repair_start_ratio: float = 0.15`
- `repair_end_ratio: float = 0.50`

### 4.5 priority 组合权重

- `priority_entropy_weight: float = 1.0`
- `priority_structure_weight: float = 0.8`
- `priority_age_weight: float = 0.4`
- `priority_confidence_weight: float = 0.6`

### 4.6 schedule 参数

- `draft_ratio: float = 0.15`
- `repair_ratio: float = 0.35`
- `finalize_ratio: float = 0.50`
- `early_sparse_min_scale: float = 0.5`
- `early_sparse_schedule_type: str = "none"`  
  可选值建议：
  - `none`
  - `linear`
  - `exp`

### 4.7 structure prior 参数

- `structure_prior_mode: str = "none"`  
  可选值建议：
  - `none`
  - `token_type`
  - `token_type_with_context`
- `structure_prior_strength: float = 1.0`

### 4.8 日志参数

- `diagnostic_log_interval: int = 1`
- `diagnostic_collect_token_events: bool = False`

---

## 5. 数据结构改造清单

在 `sample()` 与 `infill()` 内引入以下状态张量或缓存结构。

### 5.1 状态张量

建议新增：

- `tentative_mask: torch.Tensor[bool]`
- `final_mask: torch.Tensor[bool]`

说明：

- `MASK` 状态由 `~tentative_mask & ~final_mask & (x == mask_id)` 隐式表示
- 不建议把 `TENTATIVE` 编码成特殊 token id

### 5.2 tentative 元数据

每个位置建议维护：

- `tentative_token_ids`
- `tentative_age`
- `tentative_flip_count`
- `tentative_last_top1`
- `tentative_stable_run`
- `tentative_last_conf`
- `tentative_last_margin`

第一版可直接用 tensor 维护，避免 Python dict 影响 batch 性能。

### 5.3 诊断缓存

仅在 `enable_sampler_diagnostics=True` 时维护：

- `diagnostic_tentative_enter_count`
- `diagnostic_tentative_finalize_count`
- `diagnostic_tentative_rollback_count`
- `diagnostic_token_event_log`

---

## 6. utils 层新增函数清单

建议在 [utils.py](/Users/wz/code/dllm/dllm/core/samplers/utils.py) 中新增辅助函数。

### 6.1 priority 相关

- `get_top1_margin(logits) -> torch.Tensor`
- `build_structure_prior_scores(tokenizer, x, candidate_mask, mode, strength) -> torch.Tensor`
- `build_priority_scores(entropy, confidence, age_bonus, structure_scores, weights...) -> torch.Tensor`

要求：

- `build_structure_prior_scores` 在 `mode="none"` 时直接返回全零张量
- `build_priority_scores` 必须纯函数化，方便单测和 ablation

### 6.2 tentative 状态更新相关

- `update_tentative_stats(...)`
- `compute_tentative_finalize_mask(...)`
- `compute_tentative_rollback_mask(...)`

要求：

- finalize 和 rollback 判据拆开，便于单独测试

### 6.3 schedule 相关

- `get_phase(step_idx, effective_steps, draft_ratio, repair_ratio) -> str`
- `scale_quota_for_schedule(base_quota, step_idx, effective_steps, schedule_type, min_scale) -> quota`
- `split_step_budgets(total_budget, tentative_ratio, repair_ratio) -> (k_conf, k_tent, k_repair)`

要求：

- 如果 `enable_early_sparse_quota=False`，则 `scale_quota_for_schedule` 返回原 quota

---

## 7. sampler 主流程改造清单

## 7.1 第一步：保留 baseline 主干

在 [mdlm.py](/Users/wz/code/dllm/dllm/core/samplers/mdlm.py) 中，先保证：

- 当所有 `enable_*` 参数都为默认值时
- `sample()` 和 `infill()` 输出与当前实现一致

这是第一优先级。

## 7.2 第二步：插入 tentative 状态更新逻辑

在每个 step 的 `logits / probs / entropy / confidence` 算完之后，先执行：

- 更新已有 `TENTATIVE` 的统计
- 根据条件把一部分 tentative 转为 `FINAL`
- 根据条件把一部分 tentative 回滚为 `MASK`

这一段建议单独整理成小块代码，避免与 baseline 主路径混在一起。

## 7.3 第三步：保留普通高置信 finalize 通道

普通高置信 token 仍沿用当前主逻辑：

- confidence-first
- 直接 finalize

要求：

- 只有当 tentative 相关开关开启时，才额外从总 quota 中切出 `K_tent`
- 否则全部 quota 仍由 baseline 使用

## 7.4 第四步：新增 tentative 通道

在 early phase 内：

- 用 priority 分数在 `candidate_mask` 上选少量位置
- 这些位置不直接 finalize
- 而是写入 token 到 `x`
- 同时标记为 `tentative_mask=True`

要求：

- `enable_tentative_commit=False` 时，这条通道完全不执行

## 7.5 第五步：新增 repair 通道

仅当 `enable_repair_stage=True` 时启用：

- repair phase 内允许极少量位置进入修复池
- 第一版建议只处理 tentative
- 第二版再考虑极少量 low-confidence final

---

## 8. 推荐实现顺序

## 阶段 A：最小止损版

目标：

- 最快验证“掉点是不是 mainly 来自 hard commit”

实现项：

- `enable_tentative_commit`
- `enable_targeted_remask`
- `tentative_*` 判据参数
- `enable_sampler_diagnostics`

这一步先不做：

- structure priority
- early sparse quota
- final repair

## 阶段 B：结构感知版

目标：

- 验证“纯熵误选”是不是第二个主要问题

实现项：

- `enable_structure_priority`
- `structure_prior_mode`
- `priority_*_weight`
- `enable_priority_age_bonus`

## 阶段 C：schedule 版

目标：

- 验证“early stage 少写”是否进一步减少误锁

实现项：

- `enable_early_sparse_quota`
- `early_sparse_schedule_type`
- `early_sparse_min_scale`
- `draft_ratio / repair_ratio / finalize_ratio`

## 阶段 D：repair 扩展版

目标：

- 允许极少量 final token 参与修复

实现项：

- `enable_repair_stage`
- `repair_final_budget_ratio`
- `remask_only_tentative=False`

---

## 9. 消融实验映射

为保证每个模块可独立验证，建议按下面方式组织实验。

### 9.1 baseline

```text
enable_entropy_priority=False
enable_tentative_commit=False
enable_targeted_remask=False
enable_structure_priority=False
enable_early_sparse_quota=False
enable_repair_stage=False
```

### 9.2 验证 hard commit 问题

```text
enable_entropy_priority=True
enable_tentative_commit=False
```

对比：

```text
enable_entropy_priority=True
enable_tentative_commit=True
enable_targeted_remask=True
```

### 9.3 验证 structure prior 价值

在 tentative + remask 基础上，对比：

```text
enable_structure_priority=False
```

vs

```text
enable_structure_priority=True
structure_prior_mode=token_type
```

### 9.4 验证 early sparse quota

在前述最佳设置上，对比：

```text
enable_early_sparse_quota=False
```

vs

```text
enable_early_sparse_quota=True
early_sparse_schedule_type=exp
```

### 9.5 验证 repair stage

最后再对比：

```text
enable_repair_stage=False
```

vs

```text
enable_repair_stage=True
repair_final_budget_ratio > 0
```

---

## 10. 日志与可观测性清单

建议所有诊断都受 `enable_sampler_diagnostics` 控制。

必须记录：

- 每步 tentative 数量
- 每步 tentative finalize 数量
- 每步 tentative rollback 数量
- 每步普通 finalize 数量
- 每步 quota 拆分结果

推荐记录：

- tentative 生命周期统计
- 按 token 类型分桶的 tentative 成功率
- rollback / tentative 比率

可选记录：

- token 级事件日志
- 位置级 flip_count 分布

---

## 11. 单元测试建议

建议在 `/Users/wz/code/dllm/scripts/tests/test_sampling_utils.py` 补以下测试。

- priority 公式在各模块关闭时退化正确
- tentative finalize 判据正确
- tentative rollback 判据正确
- quota schedule 在 `enable_early_sparse_quota=False` 时不改变 baseline
- structure prior 在 `mode=none` 时返回零分

第一版不必覆盖完整 sampler，只要把关键纯函数测住即可。

---

## 12. 最终交付顺序

推荐按下面顺序提交工程改动。

1. 配置参数骨架
2. tentative 状态与诊断
3. targeted remask
4. structure-aware priority
5. early sparse quota
6. repair stage
7. 单测与样例命令更新

---

## 13. 一句话实施原则

实现上最重要的一条是：

> 每个新增能力都必须有独立参数开关，并且在关闭时严格退化回当前 baseline。

这样你后面的所有结论才能真正可解释、可复现、可做消融。
