# MDLM 当前采样器三块核心逻辑说明

本文档整理当前 `/Users/wz/code/dllm/dllm/core/samplers/mdlm.py` 中已经落地的三块核心逻辑：

1. `credit-based entropy priority`
2. `tentative + targeted remask`
3. `structure priority`

目标是帮助你在做消融和调参时，快速判断“当前到底启用了什么”、“每一块逻辑什么时候生效”以及“它们之间如何配合”。

---

## 1. Credit-Based Entropy Priority

### 1.1 目标

这块逻辑负责回答一个问题：

- **什么时候允许新的高熵位置优先进入解码流程**

它替换了之前的：

- `entropy_min_tokens_per_step`
- “每步固定至少处理几个高熵 token”

当前实现改成了：

- **credit-based sparse trigger**

也就是：

- 不是每一步都强行处理高熵位置
- 而是在 early phase 内逐步累积 `entropy_credit`
- 当 credit 足够且存在合格候选时，才在某一步稀疏地发放 `1` 个 entropy 名额

### 1.2 触发阶段

当前通过三段式 phase 控制：

- `entropy_warmup_ratio`
- `entropy_active_end_ratio`
- `entropy_end_ratio`

默认含义是：

- `0% ~ 5%`：warmup，不发 entropy 名额
- `5% ~ 20%`：active，按 `entropy_credit_rate` 正常累积 credit
- `20% ~ 30%`：cooldown，按 `0.5 * entropy_credit_rate` 累积 credit
- `30%` 之后：关闭，不再新增 entropy 名额

具体由 `/Users/wz/code/dllm/dllm/core/samplers/utils.py` 中的：

- `compute_entropy_phase_scale(...)`

负责。

### 1.3 Credit 累积方式

每个样本维护一份：

- `entropy_credit: float tensor [B]`

每一步如果仍处于 entropy 有效阶段，就执行：

- `entropy_credit += phase_scale * entropy_credit_rate`

具体由：

- `update_entropy_credit(...)`

负责。

### 1.4 什么时候真正发放 entropy 名额

只有同时满足以下条件时，当前步才会真正发放 entropy 名额：

1. `enable_entropy_priority=True`
2. `enable_entropy_credit_scheduler=True`
3. 当前 step 处于 active 或 cooldown
4. 当前样本 `entropy_credit >= 1`
5. 当前样本存在至少一个“高质量 entropy 候选”

满足后，本步：

- `entropy_trigger_count = 1`
- `entropy_credit -= 1`

否则：

- `entropy_trigger_count = 0`

当前实现限制：

- `entropy_max_trigger_per_step=1`

也就是说单步最多只发一个 entropy 名额。

### 1.5 当前这块逻辑的本质

一句话说：

- **当前 entropy priority 不再是“每步固定抢一个高熵位”，而是“在前期少数合适的步上，稀疏地发放一个高熵优先名额”。**

---

## 2. Tentative + Targeted Remask

### 2.1 目标

这块逻辑负责回答另一个问题：

- **高优先级位置是不是应该一被选中就直接冻结？**

当前答案是否定的。

所以现在把状态拆成：

- `MASK`
- `TENTATIVE`
- `FINAL`

高优先级位置如果走的是 tentative 通道，不会立刻变成最终 token，而是先变成：

- `TENTATIVE`

### 2.2 Tentative 的语义

`TENTATIVE` 的含义是：

- 当前 token 先写进上下文，允许后续位置看到它
- 但它还没有真正冻结
- 后续几步会继续观察它是否稳定

也就是说，这块逻辑把两件事拆开了：

- 让 token 尽早进入上下文
- 让 token 尽早不可逆冻结

### 2.3 Tentative 的观察指标

当前每个 tentative 位置会维护：

- `tentative_token_ids`
- `tentative_age`
- `tentative_flip_count`
- `tentative_last_top1`
- `tentative_stable_run`
- `tentative_last_conf`
- `tentative_last_margin`

这些指标会在每一步更新，用来判断：

- 它是不是已经足够稳定
- 它是不是应该被 remask 回去

更新逻辑在：

- `update_tentative_stats(...)`

### 2.4 Finalize 条件

当前 tentative 位置满足以下任一类条件时会转成 `FINAL`：

- 持有时间足够长，且连续稳定步数足够
- 当前置信度足够高
- 当前 top1-top2 margin 足够大

对应函数：

- `compute_tentative_finalize_mask(...)`

### 2.5 Rollback 条件

如果开启：

- `enable_targeted_remask=True`

那么 tentative 位置满足以下任一条件时会被 remask 回 `MASK`：

- 持有时间过长但仍不稳定
- `flip_count` 过高
- 当前置信度过低

对应函数：

- `compute_tentative_rollback_mask(...)`

### 2.6 “targeted” 的含义

这里不是全局 remask，也不是对所有已生成 token 回滚。

当前实现只会定向处理：

- 已经进入 `TENTATIVE` 的位置

不会对 `FINAL` token 做 repair/remask。

### 2.7 当前这块逻辑的本质

一句话说：

- **它让高风险位置可以更早参与推理，但保留后悔机会。**

---

## 3. Structure Priority

### 3.1 目标

这块逻辑负责改进一个问题：

- **高熵不等于骨架**

也就是说，单纯按 entropy 排序，会把很多“只是暂时不确定”的位置也提到前面。

所以当前代码给一部分更像“结构/逻辑骨架”的 token 额外加分。

### 3.2 当前覆盖范围

当前 `token_type` 结构先验只覆盖两类：

- 逻辑连接词
- 代码关键词

例如：

- `therefore`
- `thus`
- `since`
- `given`
- `now`
- `to`
- `return`
- `if`
- `else`
- `for`
- `while`

并且当前实现做了：

- 大小写不敏感
- 去 tokenizer 前缀
- 去两端常见标点

所以像：

- `Therefore`
- `therefore,`
- `Thus:`

都会命中。

### 3.3 当前两种模式

#### `token_type`

只看当前位置预测 token 本身是否命中结构词表。

如果命中：

- 加基础分 `structure_prior_strength`

#### `token_type_with_context`

除了看当前位置 token 本身，还会看它邻近位置是否已经是结构 token。

如果相邻位置带有结构 token，会再给一个较小 bonus。

### 3.4 当前打分方式

结构分数不是单独决策，而是进入 tentative 候选总分：

- `priority = entropy + structure + age - confidence`

更准确地说，当前是带权重版本：

- `priority_entropy_weight * entropy`
- `+ priority_structure_weight * structure`
- `+ priority_age_weight * age`
- `- priority_confidence_weight * confidence`

所以 `structure priority` 的作用是：

- 不是直接替代 entropy
- 而是在 entropy 候选之间，帮助系统更偏向“更像骨架”的位置

### 3.5 它什么时候生效

当前 `structure priority` 不是独立通道。

它只会在以下条件同时满足时真正起作用：

1. `enable_entropy_priority=True`
2. `enable_entropy_credit_scheduler=True`
3. 当前步实际发放了 entropy 名额
4. `enable_tentative_commit=True`
5. `enable_structure_priority=True`

所以它本质上是：

- **建立在 entropy-triggered tentative 通道上的排序增强项**

而不是单独的一条选位逻辑。

### 3.6 当前这块逻辑的本质

一句话说：

- **当前 structure priority 是一个轻量启发式加分器，用来让 entropy 通道更偏向逻辑连接词和代码关键词。**

---

## 4. 三块逻辑如何配合

可以把当前整体流程理解成这样：

1. baseline 通道仍然存在  
   按原本的 confidence-first 方式，处理大多数普通位置。

2. entropy priority 决定“什么时候插入高优先级位”  
   不是每步都插，而是通过 credit 稀疏触发。

3. structure priority 决定“在这些高优先级位里，更偏向谁”  
   它不独立发名额，只参与 entropy 候选排序。

4. tentative/remask 决定“这些高优先级位是不是立刻冻结”  
   不是立刻 freeze，而是先 tentative，再根据稳定性 finalize 或 rollback。

所以当前的关系不是三条完全独立的分支，而是：

- `entropy priority`：决定是否发名额
- `structure priority`：决定名额更偏向哪些位置
- `tentative + targeted remask`：决定这些位置怎么提交、怎么回滚

---

## 5. 当前最关键的实验含义

基于现有代码，做实验时可以这样理解：

### Baseline

- 只走 confidence-first
- 不发 entropy 名额
- 不走 tentative

### Entropy-Only

- 发 entropy 名额
- 但不走 tentative
- 命中的 entropy 候选直接写回

### Tentative + Targeted Remask

- 发 entropy 名额
- 命中的 entropy 候选进入 `TENTATIVE`
- 后续可能 finalize，也可能 rollback

### Structure Priority

- 不是单独模式
- 而是在 tentative entropy 通道上再加结构偏置

---

## 6. 当前推荐重点关注的参数

如果你后面要调当前实现，优先关注这些参数：

### Entropy Trigger

- `enable_entropy_priority`
- `enable_entropy_credit_scheduler`
- `entropy_credit_rate`
- `entropy_warmup_ratio`
- `entropy_active_end_ratio`
- `entropy_end_ratio`
- `entropy_top_k`
- `entropy_use_quality_gate`
- `entropy_conf_floor`
- `entropy_age_threshold`
- `entropy_top_candidate_pool`

### Tentative / Remask

- `enable_tentative_commit`
- `enable_targeted_remask`
- `tentative_min_hold_steps`
- `tentative_stable_steps`
- `tentative_max_hold_steps`
- `tentative_final_prob_thresh`
- `tentative_final_margin_thresh`
- `remask_rollback_prob_thresh`
- `remask_flip_thresh`

### Structure Priority

- `enable_structure_priority`
- `enable_priority_age_bonus`
- `structure_prior_mode`
- `structure_prior_strength`
- `priority_entropy_weight`
- `priority_structure_weight`
- `priority_age_weight`
- `priority_confidence_weight`

---

## 7. 一句话总结

当前代码的整体思路可以压缩成一句话：

- **通过 credit 稀疏地在 early phase 插入少量高熵候选，用结构先验帮助挑更像骨架的位置，再用 tentative/remask 避免这些高风险位置被过早硬冻结。**
