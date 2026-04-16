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
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
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
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
  --entropy_top_k 64 \
  --tentative_budget_ratio 0.1 \
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
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --enable_structure_priority True \
  --enable_priority_age_bonus True \
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
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
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],entropy_min_tokens_per_step=0" \
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
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64" \
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
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_tentative_commit=True,enable_targeted_remask=True,entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64,tentative_budget_ratio=0.1,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3" \
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
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_tentative_commit=True,enable_targeted_remask=True,enable_structure_priority=True,enable_priority_age_bonus=True,entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64,structure_prior_mode=token_type_with_context,structure_prior_strength=1.0,tentative_budget_ratio=0.1,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-structure-priority-gpu3.log
```

## 10. 如果你想保留日志

建议先建日志目录：

```bash
mkdir -p /disk/wangzhe/dllm/.logs
```

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
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
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
  --enable_tentative_commit True \
  --enable_targeted_remask True \
  --enable_structure_priority True \
  --enable_priority_age_bonus True \
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
  --entropy_top_k 64 \
  --structure_prior_mode token_type_with_context

accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],entropy_min_tokens_per_step=0"

accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],enable_entropy_priority=True,enable_tentative_commit=True,enable_targeted_remask=True,enable_structure_priority=True,enable_priority_age_bonus=True,entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64,structure_prior_mode=token_type_with_context,structure_prior_strength=1.0,tentative_budget_ratio=0.1,tentative_min_hold_steps=1,tentative_stable_steps=2,tentative_max_hold_steps=3"
```

如果这几步都正常，再继续做更大的 benchmark。
