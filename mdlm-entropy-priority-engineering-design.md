# MDLM Entropy-Priority 优化工程设计稿

## 1. 背景

当前项目在原始 `MDLM` 推理器上实现了一版 **Entropy-Priority**：

- 在每个 block 的早期步骤中；
- 不再只按置信度提交 token；
- 而是给一部分高熵位置保留更新配额；
- 以避免“关键骨架位被一直拖到最后”。

这个方向的动机是合理的：

- 原始 confidence-first arbitrary-order 容易绕过高不确定性位置；
- 这些位置往往对应逻辑骨架、结构分叉、关键连接点；
- 如果它们被长期延后，会出现 entropy degradation，导致推理空间过早坍缩。

但当前版本掉点严重，说明问题不在“早关注骨架”这个方向本身，而在于当前实现方式过硬、过粗。

---

## 2. 当前问题总结

### 2.1 问题一：早介入被错误实现成了早冻结

当前实现是：

> early select + hard commit

也就是高熵位置一旦被选中，就直接写回离散 token，并退出迭代。

这会带来两个后果：

- 高风险位置在上下文还不充分时被过早锁死；
- 一旦预测错误，后续几乎无法修正。

所以现在真正有问题的不是：

- 早碰高熵位置

而是：

- 过早把高熵位置 finalize

### 2.2 问题二：纯熵不能精确刻画“骨架”

当前策略默认：

> 高熵位置 = 优先处理的位置

但在真实任务中，高熵位置不一定是逻辑骨架，也可能只是：

- 数字
- 实体
- 变量
- 局部细节
- 暂时模糊但不影响整体结构的 token

所以熵是一个有价值的“发现信号”，但不是一个足够准确的“骨架定义”。

---

## 3. 优化目标

下一版优化的目标，不是继续微调 `entropy_min_tokens_per_step` 这类表层超参，而是把策略从：

> early select + hard commit

升级成：

> early select + tentative commit + delayed finalize

并且把优先级从：

> entropy-only

升级成：

> entropy + structure-aware prior + age bonus + confidence penalty

换句话说，下一版系统要同时解决两个问题：

1. **让骨架更早进入上下文**
2. **避免骨架过早不可逆冻结**

---

## 4. 设计原则

### 原则一：早介入，不等于早 finalize

高优先级位置应该更早被“看见”，但不应该更早被“锁死”。

### 原则二：熵只负责发现不确定性，不负责定义骨架

真正的优先级应由多个信号共同决定。

### 原则三：早期保守，中期修错，后期收敛

推理过程应采用分阶段设计，而不是全程使用同一套 aggressive 策略。

### 原则四：先做最小可行修复，再考虑 soft token

第一阶段优先做：

- tentative state
- delayed finalize
- targeted remask

而不是一开始就引入 soft token / soft embedding。

---

## 5. 总体方案

## 5.1 核心方向

借鉴两类经验：

- **ReMDM**：已提交 token 可按条件 remask，再次参与解码
- **eMIGM / coarse-to-fine**：early stage 少写一些 token，尤其少做高风险提交

对应到当前项目，形成下面这套主方案：

> **Structure / Entropy-aware Tentative Commit + Targeted ReMask + Slow Early Quota**

即：

- 早期总写入量更保守；
- 高优先级骨架候选更早进入上下文；
- 但先进入 `TENTATIVE`，不是 `FINAL`；
- 如果后续不稳定，则 remask；
- 稳定后再 finalize；
- 后期关闭探索，快速收敛。

---

## 6. 状态设计

建议把 token 状态从当前的 2 态：

- `MASK`
- `FINAL`

扩展成 3 态：

- `MASK`
- `TENTATIVE`
- `FINAL`

说明：

- `MASK`：仍未提交
- `TENTATIVE`：当前 token 已进入上下文，但允许撤销
- `FINAL`：真正冻结，不再参与修正

可选：

- 不额外引入 `REMASKED` 作为正式状态
- 只通过事件日志记录该位置是否经历过 rollback

这样状态机更简单，调试也更清晰。

---

## 7. 三阶段推理流程

## 7.1 阶段一：Draft

目标：

- 形成一个初步但保守的结构草稿

策略：

- 总写入量较小
- entropy-priority 开启，但预算极小
- 被选中的高优先级位置进入 `TENTATIVE`
- 普通高置信位置仍可直接 `FINAL`
- 暂不 remask `FINAL`

直觉：

- 先让关键骨架被上下文“看见”
- 但不要过早不可逆确定

## 7.2 阶段二：Repair

目标：

- 利用更丰富的上下文纠正前期高风险位置

策略：

- 保持中等写入量
- 对 `TENTATIVE` 开启 targeted remask
- 可选地允许极少量低置信 `FINAL` 进入修复池
- 不再大规模新增 entropy-priority 候选

直觉：

- 让系统有一次“修错窗口”
- 主动修复早期 tentative 决策

## 7.3 阶段三：Finalize

目标：

- 停止探索，快速收敛

策略：

- 提高总写入量
- 关闭新增 tentative
- 关闭或极大降低 remask
- 只做 finalize

直觉：

- 中后期不再制造新的不稳定性

---

## 8. Priority 设计

当前项目的关键升级点之一，是把“纯熵选位”升级为“结构感知优先级”。

建议使用如下组合分数：

```text
priority_i =
  α * H_i
+ β * S_i
+ γ * A_i
- δ * C_i
```

其中：

- `H_i`：归一化熵，负责发现不确定性
- `S_i`：结构先验，负责判断该位置是否更可能是骨架
- `A_i`：age bonus，表示该位置高风险但长期未被处理
- `C_i`：置信度或 margin，越高说明越不需要 tentative 通道

### 8.1 结构先验 `S_i`

第一版建议先做轻量结构先验，而不是复杂额外模型。

可包含：

- 数学：`= + - * / ( ) : , therefore thus so`
- 代码：`def if else for while return ( ) { } : ,`
- CoT：`because therefore however so then first next finally`

实现建议：

- 对结构符号类 token 直接给较强 prior
- 对连接词类 token 给较弱 prior
- 后续再考虑 span 级结构先验

### 8.2 年龄奖励 `A_i`

用于避免“高风险位置永远拖着不碰”。

可简单定义为：

- 连续若干步保持高熵但未被处理，则 priority 缓慢上升

### 8.3 置信度惩罚 `C_i`

若某位置当前已经很自信，则不应进入 tentative 通道。

作用：

- 降低普通高置信 token 与 tentative 通道的冲突

---

## 9. Scheduler 设计

除了优先级本身，另一条主优化线是：

> early stage 少写一些 token

建议使用“双慢启动”调度。

## 9.1 总写入量 `K_total(t)`

早期少，后期多。

目标：

- 前期只形成雏形
- 中期修正
- 后期收敛

推荐第一版：

- 保持原有总 quota 逻辑不大改
- 只在其上额外控制 tentative 子预算

确认 tentative/remask 有价值后，再尝试把总 quota 改为更保守的 exp-like increasing schedule。

## 9.2 entropy / tentative 子预算 `K_tent(t)`

这部分要比总预算更保守。

建议：

- 只在前 `10%~20%` steps 打开
- 每步最多占总 quota 的 `5%~10%`
- 中后期停止新增 tentative

原因：

- 目标只是“避免骨架位一直被拖到最后”
- 不是全程让高熵位置抢占预算

---

## 10. Tentative / Finalize / ReMask 判据

## 10.1 对每个 tentative 位置维护

- `token_t`：当前 top-1 token
- `p_t`：当前 top-1 概率
- `margin_t = p1 - p2`
- `age_t`：进入 tentative 后经过的步数
- `flip_count_t`：top-1 token 变化次数

## 10.2 Finalize 条件

推荐用偏保守组合：

- `age_t >= min_hold_steps`
- 且连续 `stable_steps` top-1 token 不变
- 或 `p_t >= final_prob_thresh`
- 或 `margin_t >= final_margin_thresh`

## 10.3 ReMask 条件

推荐：

- `age_t >= max_hold_steps` 但仍不稳定
- `flip_count_t >= flip_thresh`
- `p_t <= rollback_prob_thresh`
- token 在短时间内反复跳变

第一版建议：

- 只对 `TENTATIVE` 开启 remask
- 不直接 remask 普通 `FINAL`

---

## 11. 推荐的最小可行实现版本

这是建议优先落地的止损版：

### 状态

- `MASK`
- `TENTATIVE`
- `FINAL`

### 行为

- 只在前 `15%~20%` steps 开 tentative
- 每步 tentative 子预算只占总 quota 的 `5%~10%`
- priority 使用：
  - entropy
  - structure prior
  - age bonus
  - confidence penalty
- `TENTATIVE` 持有 `1~3` 步
- 不稳定则 remask
- 稳定后才 finalize
- 暂不 remask 普通 final token

### 默认超参建议

```text
draft_ratio              = 0.15
repair_ratio             = 0.35
finalize_ratio           = 0.50

tentative_budget_ratio   = 0.05 ~ 0.10
min_hold_steps           = 1
stable_steps             = 2
max_hold_steps           = 3
final_prob_thresh        = 0.82
final_margin_thresh      = 0.35
rollback_prob_thresh     = 0.45
flip_thresh              = 2
```

---

## 12. 最小状态机伪代码

```python
for step in steps:
    logits = model(x_t)
    probs = softmax(logits)
    entropy = compute_entropy(probs)

    # 1. 更新已有 tentative token
    for i in tentative_positions:
        update_history(i, probs[i])
        if should_finalize(i):
            state[i] = FINAL
        elif should_rollback(i):
            state[i] = MASK

    # 2. 普通高置信位置 -> FINAL
    conf_candidates = select_confident_masked_positions(...)
    finalize_topk(conf_candidates, K_conf)

    # 3. 早期高优先级位置 -> TENTATIVE
    if step < early_phase_steps:
        prio_candidates = select_masked_positions_by_priority(...)
        make_tentative(prio_candidates, K_tent)

    # 4. 构造下一步输入
    # FINAL: 使用冻结 token
    # TENTATIVE: 使用当前 token，但允许后续回滚
    # MASK: 继续 mask
    x_t = rebuild_sequence(...)
```

---

## 13. 实验验证顺序

不要一开始就把所有改动一起打开。

建议按下面顺序做 ablation。

## 13.1 第一轮：验证主要问题是否来自 hard commit

比较：

1. baseline
2. entropy-priority + hard commit
3. entropy-priority + tentative + remask

目标：

- 验证掉点主因是否真的是“不可逆早提交”

## 13.2 第二轮：验证结构先验是否必要

比较：

1. entropy-only priority
2. entropy + structure priority

目标：

- 验证“纯熵误选”是否是当前第二个主要问题

## 13.3 第三轮：验证 early-sparse schedule 是否更稳

比较：

1. 原始 quota
2. 保守 early quota
3. exp-like increasing quota

目标：

- 验证 early stage 少写是否能进一步减少误锁

## 13.4 第四轮：再决定是否扩展到 remask final / soft token

只有当前三轮都说明方向有效后，再考虑：

- 对极少量 `FINAL` 做 repair remask
- soft token / soft state

---

## 14. 必须记录的日志

## 14.1 tentative 生命周期

对每个 tentative token 记录：

- 进入 tentative 的 step
- 持续步数
- 最终是 finalize 还是 rollback
- top-1 变化次数

## 14.2 rollback 率

按 step 记录：

- tentative 数量
- rollback 数量
- rollback / tentative 比值

## 14.3 tentative 成功率

统计：

- tentative token 最终是否保持原 token finalize
- 该 tentative 是否对最终答案有正贡献

## 14.4 按 token 类型分桶

建议至少区分：

- 结构符号
- 连接词
- 数字
- 实体 / 内容词

这有助于快速定位：

- 究竟是骨架位在受益
- 还是数字/细节位在拖后腿

---

## 15. 研发推进顺序

推荐按下面顺序推进：

### 第一步：止损版

- tentative state
- delayed finalize
- targeted remask on tentative
- 小 quota、early-only

### 第二步：结构感知版

- token-type prior
- context boundary prior
- age bonus

### 第三步：更完整的 repair

- 极少量 low-confidence final 进入 repair
- 更细致的 remask schedule

### 第四步：soft state / soft token（后置）

只有在前面三步已经证明方向有效时，再考虑：

- top-k soft mixture
- masked embedding + expected embedding
- delayed sharpening

---

## 16. 一句话总结

当前优化方向的核心，不是继续讨论“高熵是不是骨架”，而是解决一个更本质的问题：

> 在离散扩散式文本生成里，**什么时候一个 token 应该被上下文看见**，以及 **什么时候它应该被系统锁定**。

原始实现把这两件事绑定在了一起。

下一版设计的核心任务，就是把它们解绑：

> **只让高优先级位置更早进入上下文，不让它们更早不可逆冻结。**
