# MDLM 测试运行说明

本文档说明如何在 `/disk/wangzhe/dllm` 中运行当前项目的测试，重点覆盖：

- Python 单元测试
- 本次 `MDLM Entropy-Priority` 改动相关的回归测试
- 服务器上的小样本推理验证

本文默认你会在服务器环境中执行命令。

## 1. 环境准备

每次运行测试前，建议先执行：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH
```

如果你的服务器上 `conda activate` 不能直接使用，可以先执行：

```bash
source ~/.zshrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH
```

先做一个最小检查：

```bash
python -V
pytest --version
```

## 2. 测试目录

当前仓库使用 `pytest`，测试目录由 `/disk/wangzhe/dllm/pyproject.toml` 指定为：

- `/disk/wangzhe/dllm/scripts/tests`

和这次改动最相关的测试文件是：

- `/disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py`

## 3. 先跑最关键的单测

如果你只想先验证本次 `MDLM sampler` 改动是否基本正常，优先跑：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -v
```

如果你只想跑某一类测试，可以加 `-k`：

```bash
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -k tentative -v
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -k structure -v
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -k rollback -v
```

如果你只想快速看结果，可以用：

```bash
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -q
```

## 4. 运行全部 Python 单元测试

如果你想跑整个仓库当前配置下的 Python 测试：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH
pytest
```

或者显式指定测试目录：

```bash
pytest /disk/wangzhe/dllm/scripts/tests -v
```

## 5. 推荐的回归测试顺序

建议按下面顺序跑，这样更容易定位问题：

1. `test_sampling_utils.py`
2. 其余轻量测试
3. 全量 `pytest`
4. 小样本推理验证

一个更稳妥的顺序示例：

```bash
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -v
pytest /disk/wangzhe/dllm/scripts/tests/test_schedulers.py -v
pytest /disk/wangzhe/dllm/scripts/tests/test_eval_base.py -v
pytest /disk/wangzhe/dllm/scripts/tests -v
```

## 6. GPU 环境检查

如果你接下来要做小样本推理验证，建议先确认 GPU 和 PyTorch 正常：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device="cuda")
    y = x @ x.T
    print("ok", y.shape, y.device)
PY
```

如果你想固定单卡：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 7. 先做一次语法级检查

如果你只想快速确认这次改动的几个核心文件能被 Python 正常解析：

```bash
python -m py_compile \
  /disk/wangzhe/dllm/dllm/core/samplers/utils.py \
  /disk/wangzhe/dllm/dllm/core/samplers/mdlm.py \
  /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py
```

## 8. MDLM 小样本验证

单元测试通过后，再做一轮最小推理验证。

先设置模型路径：

```bash
MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "$MODEL_PATH"
```

### 8.1 Baseline

```bash
python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --entropy_min_tokens_per_step 0
```

### 8.2 只开 entropy priority，不开 tentative

```bash
python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --enable_entropy_priority True \
  --enable_entropy_credit_scheduler True \
  --entropy_credit_rate 0.35 \
  --entropy_warmup_ratio 0.05 \
  --entropy_active_end_ratio 0.20 \
  --entropy_end_ratio 0.30 \
  --entropy_top_k 64
```

### 8.3 开 tentative + targeted remask

```bash
python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --enable_entropy_priority True \
  --enable_entropy_credit_scheduler True \
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --entropy_credit_rate 0.35 \
  --entropy_warmup_ratio 0.05 \
  --entropy_active_end_ratio 0.20 \
  --entropy_end_ratio 0.30 \
  --entropy_top_k 64 \
  --tentative_min_hold_steps 1 \
  --tentative_stable_steps 2 \
  --tentative_max_hold_steps 3
```

### 8.4 在 8.3 基础上再开 structure priority

```bash
python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --enable_entropy_priority True \
  --enable_entropy_credit_scheduler True \
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --enable_structure_priority True \
  --enable_priority_age_bonus True \
  --entropy_credit_rate 0.35 \
  --entropy_warmup_ratio 0.05 \
  --entropy_active_end_ratio 0.20 \
  --entropy_end_ratio 0.30 \
  --entropy_top_k 64 \
  --structure_prior_mode token_type_with_context \
  --structure_prior_strength 1.0
```

## 9. 指定 `gsm8k` 数据集并只跑 20 条

如果你想在评测链路里指定 `gsm8k`，这里应当使用仓库当前的任务名：

- `gsm8k_cot`

也就是通过 `/disk/wangzhe/dllm/dllm/pipelines/llada/eval.py` 跑 `lm-eval` 时，推荐命令如下。

如果你正好有 4 张卡，最省时间的方式是：

- 先执行一次公共环境初始化
- 再分别把下面 4 条命令放到 4 个 shell 中运行
- 或者在同一个 shell 里分别加 `nohup` 和 `&`

建议先创建日志目录：

```bash
mkdir -p /disk/wangzhe/dllm/.logs
```

下面这组默认值比之前更适合当前实现：

- `max_new_tokens=512`
- `steps=64`
- `block_size=32`
- `entropy_credit_rate=0.35`
- `entropy_warmup_ratio=0.05`
- `entropy_active_end_ratio=0.20`
- `entropy_end_ratio=0.30`

原因：

- 新实现不再使用“每步固定至少处理几个 entropy token”，而是用 `credit` 稀疏触发
- `steps=64` 可以显著加快评测，同时让 entropy credit 和 tentative 通道真正参与
- `block_size=32` 先保持不变，避免一次改太多变量
- `0.05 / 0.20 / 0.30` 这组 phase 配置对应 warmup / active / cooldown / off，适合作为中等触发默认值
- `entropy_credit_rate=0.35` 大致会形成“不是每步都触发，但前期会稳定插入 entropy 名额”的节奏

补充说明：

- `entropy_min_tokens_per_step` 和 `tentative_budget_ratio` 现在只保留作兼容字段，不再决定主逻辑
- 新实验请优先调 `entropy_credit_rate`、`entropy_warmup_ratio`、`entropy_active_end_ratio`、`entropy_end_ratio`

### 9.1 Baseline：`gsm8k_cot --limit 20`

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],entropy_min_tokens_per_step=0,save_generation_records_path=/disk/wangzhe/dllm/.logs/gsm8k-baseline-gpu0.jsonl,save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-baseline-gpu0.log
```

### 9.2 只开 entropy priority：`gsm8k_cot --limit 20`

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_entropy_credit_scheduler=True,entropy_credit_rate=0.35,entropy_warmup_ratio=0.05,entropy_active_end_ratio=0.20,entropy_end_ratio=0.30,entropy_top_k=64,save_generation_records_path=/disk/wangzhe/dllm/.logs/gsm8k-entropy-only-gpu1.jsonl,save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-entropy-only-gpu1.log
```

### 9.3 开 tentative + targeted remask：`gsm8k_cot --limit 20`

```bash
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_entropy_credit_scheduler=True,enable_tentative_commit=True,enable_targeted_remask=True,entropy_credit_rate=0.35,entropy_warmup_ratio=0.05,entropy_active_end_ratio=0.20,entropy_end_ratio=0.30,entropy_top_k=64,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3,save_generation_records_path=/disk/wangzhe/dllm/.logs/gsm8k-tentative-remask-gpu2.jsonl,save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-tentative-remask-gpu2.log
```

### 9.4 再加 structure priority：`gsm8k_cot --limit 20`

```bash
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_entropy_credit_scheduler=True,enable_tentative_commit=True,enable_targeted_remask=True,enable_structure_priority=True,enable_priority_age_bonus=True,entropy_credit_rate=0.35,entropy_warmup_ratio=0.05,entropy_active_end_ratio=0.20,entropy_end_ratio=0.30,entropy_top_k=64,structure_prior_mode=token_type_with_context,structure_prior_strength=1.0,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3,save_generation_records_path=/disk/wangzhe/dllm/.logs/gsm8k-structure-priority-gpu3.jsonl,save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-structure-priority-gpu3.log
```

## 10. 如果你想保留日志

建议先建日志目录：

```bash
mkdir -p /disk/wangzhe/dllm/.logs
```

如果你希望在评测时除了标准输出日志之外，再额外保存：

- 题目内容
- 完整 prompt
- 模型回答
- 启发式提取出的最终答案
- 可选的 step 级文本轨迹

现在可以直接在 `model_args` 中加入下面这些参数：

- `save_generation_records_path=/disk/wangzhe/dllm/.logs/xxx.jsonl`
- `save_generation_traces=True`
- `generation_trace_max_steps=64`
- `save_sampler_diagnostics=True`
- `diagnostic_collect_step_debug=True`（可选，记录 step 级 entropy 调试信息）
- `diagnostic_log_interval=1`（可选，按步记录；设成更大值可抽样）

说明：
- 现在只要设置 `save_sampler_diagnostics=True`，eval harness 会自动给 sampler 打开逐样本 diagnostics 采集。
- 不需要再额外手动传 `enable_sampler_diagnostics=True`。

例如：

```bash
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_entropy_credit_scheduler=True,enable_tentative_commit=True,enable_targeted_remask=True,entropy_credit_rate=0.35,entropy_warmup_ratio=0.05,entropy_active_end_ratio=0.20,entropy_end_ratio=0.30,entropy_top_k=64,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3,save_generation_records_path=/disk/wangzhe/dllm/.logs/gsm8k-tentative-remask-gpu2.jsonl,save_generation_traces=True,generation_trace_max_steps=64,save_sampler_diagnostics=True,diagnostic_collect_step_debug=True,diagnostic_log_interval=1"
```

生成的 `jsonl` 文件中，每一行对应一道题，包含：

- `question`
- `prompt`
- `response`
- `predicted_final_answer`
- `generation_trace`
- `sampler_diagnostics`（如果开启）

当前 `sampler_diagnostics` 会按样本保存，重点字段包括：

- `entropy_priority_effective`
- `entropy_trigger_count`
- `entropy_selected_token_count`
- `entropy_finalize_count`
- `tentative_enter_count`
- `tentative_finalize_count`
- `tentative_rollback_count`
- `baseline_finalize_count`
- `finalized_token_count`

字段语义：
- `entropy_selected_token_count`：被 entropy priority 选中的 token 数
- `entropy_finalize_count`：被 entropy priority 选中后立即最终确定的 token 数，只会在 `entropy-only` 路径出现
- `baseline_finalize_count`：走普通 confidence 路径最终确定的 token 数，不包含 entropy-only 的提前确定 token
- `tentative_finalize_count`：先进入 tentative，随后最终 finalize 的 token 数
- `finalized_token_count`：最终被确定的 token 总数，等于 `entropy_finalize_count + baseline_finalize_count + tentative_finalize_count`

如果同时打开了 token 事件采集，还会有：

- `token_events`

如果打开了 `diagnostic_collect_step_debug=True`，每条样本里还会有：

- `step_debug`

其中每一项会记录当前 step 的关键调试信息，例如：

- `step`
- `step_ratio`
- `phase_scale`
- `total_budget`
- `candidate_count`
- `quality_candidate_count`
- `selected_candidate_count`
- `credit_before`
- `credit_after_update`
- `credit_after_spend`
- `trigger_count`
- `reason`

常见 `reason` 包括：

- `phase_off`
- `no_budget`
- `no_masked_candidates`
- `no_qualified_candidates`
- `insufficient_credit`
- `triggered`

其中会记录每个触发事件对应的：

- `event`
- `step`
- `pos`

例如把关键单测结果保存下来：

```bash
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -v \
  2>&1 | tee /disk/wangzhe/dllm/.logs/test-sampling-utils.log
```

把小样本推理结果保存下来：

```bash
python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --enable_entropy_priority True \
  --enable_entropy_credit_scheduler True \
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --entropy_credit_rate 0.35 \
  --entropy_warmup_ratio 0.05 \
  --entropy_active_end_ratio 0.20 \
  --entropy_end_ratio 0.30 \
  --entropy_top_k 64 \
  2>&1 | tee /disk/wangzhe/dllm/.logs/mdlm-sample-tentative-remask.log
```

## 11. 常见建议

- 先跑 `/disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py`，不要一上来就全量 `pytest`
- 先做 `sample.py` 冒烟测试，再跑 `gsm8k_cot --limit 20`
- 做对比实验时固定 `seed`、`steps`、`block_size`、`temperature`
- 所有新能力默认应可关闭，所以如果出现异常，先退回 baseline 配置确认问题是否来自新开关

## 12. 我建议你实际执行的最小命令集合

如果你现在只是想最快验证本次改动，直接按这个顺序执行就够了：

```bash
source ~/.zshrc
conda activate ~/miniconda3/envs/dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH

pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -v

python -m py_compile \
  /disk/wangzhe/dllm/dllm/core/samplers/utils.py \
  /disk/wangzhe/dllm/dllm/core/samplers/mdlm.py \
  /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py

python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --entropy_min_tokens_per_step 0

python -u /disk/wangzhe/dllm/examples/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --enable_entropy_priority True \
  --enable_entropy_credit_scheduler True \
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --enable_structure_priority True \
  --enable_priority_age_bonus True \
  --entropy_credit_rate 0.35 \
  --entropy_warmup_ratio 0.05 \
  --entropy_active_end_ratio 0.20 \
  --entropy_end_ratio 0.30 \
  --entropy_top_k 64 \
  --structure_prior_mode token_type_with_context

accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],entropy_min_tokens_per_step=0"

accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=64,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_entropy_credit_scheduler=True,enable_tentative_commit=True,enable_targeted_remask=True,enable_structure_priority=True,enable_priority_age_bonus=True,entropy_credit_rate=0.35,entropy_warmup_ratio=0.05,entropy_active_end_ratio=0.20,entropy_end_ratio=0.30,entropy_top_k=64,structure_prior_mode=token_type_with_context,structure_prior_strength=1.0,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3"
```

如果这几步都正常，再继续做更大的 benchmark。

## 13. 双 GPU Entropy Grid Search

如果你现在有 `gpu0` 和 `gpu1` 两张卡可用，可以直接使用下面两份顺序脚本：

- [run_entropy_grid_gpu0.sh](/Users/wz/code/dllm/scripts/run_entropy_grid_gpu0.sh)
- [run_entropy_grid_gpu1.sh](/Users/wz/code/dllm/scripts/run_entropy_grid_gpu1.sh)

这两份脚本会共享同一个 sweep 根目录，总共覆盖 72 个 `entropy-only` 参数组合：

- `entropy_credit_rate`: `0.20, 0.30, 0.40`
- `entropy_warmup_ratio`: `0.00, 0.05`
- `entropy_active_end_ratio`: `0.10, 0.15, 0.20`
- `entropy_end_ratio`: `0.25, 0.30, 0.35, 0.40`
- `entropy_top_k`: 固定 `64`

其中：

- `gpu0` 跑前 36 组
- `gpu1` 跑后 36 组

两个脚本都固定使用：

- `tasks=gsm8k_cot`
- `num_fewshot=5`
- `limit=200`
- `max_new_tokens=256`
- `steps=64`
- `block_size=32`
- `cfg_scale=0.0`
- `begin_suppress_tokens=[126081;126348]`
- `save_generation_traces=True`
- `save_sampler_diagnostics=True`
- `diagnostic_collect_step_debug=True`

### 13.1 启动方式

先在两个终端里分别执行：

```bash
export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

bash /disk/wangzhe/dllm/scripts/run_entropy_grid_gpu0.sh
```

```bash
export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

bash /disk/wangzhe/dllm/scripts/run_entropy_grid_gpu1.sh
```

如果你想手动指定 sweep 目录，也可以在两个终端都先导出同一个 `OUTPUT_ROOT`：

```bash
export OUTPUT_ROOT=/disk/wangzhe/dllm/.logs/sweeps/20260419-entropy-grid72
```

### 13.2 输出目录结构

默认情况下，脚本会在下面创建一个共享目录：

```text
/disk/wangzhe/dllm/.logs/sweeps/{timestamp}-gsm8k-cot-limit200-entropy-grid72/
```

目录结构大致如下：

```text
{sweep_root}/
  sweep_config.json
  grid_manifest.csv
  leaderboard.csv
  leaderboard.md
  failures.csv
  runs/
    r001__cr0p20-wu0p00-ae0p10-ee0p25/
      command.sh
      eval.log
      generation_records.jsonl
      meta.json
      done.ok
```

每个参数组合都会落到独立 `run` 目录中，所以不会再出现日志和 `jsonl` 被覆写的问题。

### 13.3 断点续跑

脚本会自动检查每个 run 目录下是否存在 `done.ok`：

- 如果存在，就跳过这个组合
- 如果只有 `failed.ok`，再次运行脚本时会重新跑该组合

也就是说，中途中断后，直接重新执行同一份脚本即可继续。

### 13.4 如何看排行榜

每次 run 完成后，脚本都会自动刷新：

- `grid_manifest.csv`
- `leaderboard.csv`
- `leaderboard.md`
- `failures.csv`

默认排序规则是：

1. `Flexible EM` 降序
2. `Strict EM` 降序
3. `duration` 升序

你可以直接打开：

- [leaderboard.md](</disk/wangzhe/dllm/.logs/sweeps/leaderboard.md>)

实际文件会在具体的 sweep 根目录下。

### 13.5 只检查某一张 GPU 的失败项

`failures.csv` 里会包含 `assigned_gpu` 字段，所以可以直接按 GPU 过滤：

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("/disk/wangzhe/dllm/.logs/sweeps/20260419-entropy-grid72/failures.csv")
with path.open("r", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

for row in rows:
    if row["assigned_gpu"] == "0":
        print(row["run_id"], row["eval_log_path"])
PY
```

## 14. `gpu2` 单卡 48 组 Entropy Grid Search

如果你想单独在 `gpu2` 上跑一轮新的参数搜索，可以使用：

- [run_entropy_grid_gpu2_len256_step128.sh](/Users/wz/code/dllm/scripts/run_entropy_grid_gpu2_len256_step128.sh)

这轮搜索固定使用：

- `tasks=gsm8k_cot`
- `num_fewshot=5`
- `limit=100`
- `max_new_tokens=256`
- `steps=128`
- `block_size=32`
- `entropy_top_k=64`
- `gpu=2`

网格参数为：

- `entropy_credit_rate`: `0.20, 0.30, 0.40, 0.50`
- `entropy_warmup_ratio`: `0.00, 0.05`
- `entropy_active_end_ratio`: `0.10, 0.20`
- `entropy_end_ratio`: `0.25, 0.30, 0.35`

总共 `48` 组，顺序仍然是：

1. `credit_rate`
2. `warmup_ratio`
3. `active_end_ratio`
4. `end_ratio`

### 14.1 启动方式

```bash
export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

bash /disk/wangzhe/dllm/scripts/run_entropy_grid_gpu2_len256_step128.sh
```

如果你想手动指定 sweep 目录：

```bash
export OUTPUT_ROOT=/disk/wangzhe/dllm/.logs/sweeps/20260420-entropy-grid48-gpu2
```

### 14.2 sweep 目录命名

默认目录格式为：

```text
/disk/wangzhe/dllm/.logs/sweeps/{timestamp}-gsm8k-cot-limit100-len256-step128-entropy-grid48-gpu2/
```

它和之前的：

- `...limit200-entropy-grid72`

使用不同的 registry 文件，不会互相覆盖，也不会误续跑到旧实验里。

### 14.3 输出与 resume

输出结构和双 GPU 版本保持一致：

- `grid_manifest.csv`
- `leaderboard.csv`
- `leaderboard.md`
- `failures.csv`
- `runs/rXXX__.../`

`done.ok` 存在时会自动跳过，所以中断后直接重新执行同一脚本即可 resume。
