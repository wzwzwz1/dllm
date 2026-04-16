# Entropy-Priority LLaDA Test Guide

## 目标

本文档说明如何在 `/disk/wangzhe/dllm` 中，使用单张 GPU 对你设计的 entropy-priority 推理方法和基线方法做对比测试。

文档的目标不是一次性跑完整大规模评测，而是先用最小风险的方式完成：

- 环境检查
- 单样本冒烟测试
- 小样本 benchmark 对比
- 基线与新方法的公平比较

本文默认你已经满足以下条件：

- 代码仓库路径为 `/disk/wangzhe/dllm`
- conda 环境名为 `dllm`
- 模型已经下载完成
- Hugging Face 缓存目录位于 `/disk/wangzhe/.cache/huggingface`

## 模型路径

请使用下面这个已经下载好的模型目录：

```bash
/disk/wangzhe/.cache/huggingface/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
```

推荐先设置一个 shell 变量：

```bash
MODEL_PATH=/disk/wangzhe/.cache/huggingface/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "$MODEL_PATH"
```

`echo "$MODEL_PATH"` 的输出必须是一整行，不能包含换行。

如果路径中混入换行，`transformers` 会把它错误地当成 Hugging Face repo id 处理，进而报错。

## 共享服务器上的安全原则

- 不要使用 `apt`
- 不要同时启动多组测试
- 一次只使用一张 GPU
- 先用 `nvidia-smi` 确认当前机器上的 GPU 使用情况
- 如有需要，用 `export CUDA_VISIBLE_DEVICES=0` 显式绑定单卡
- 先跑 `sample.py` 冒烟测试，再跑 benchmark
- 第一轮 benchmark 只使用 `--limit 20`
- 不要一开始就测试 `HumanEval` 或 `MBPP`
- 不要一开始就并发跑多个任务

## 环境准备

每次测试前先执行：

```bash
source ~/.zshrc
source /home/wangzhe/miniconda3/etc/profile.d/conda.sh
conda activate dllm
cd /disk/wangzhe/dllm
export PYTHONPATH=.:$PYTHONPATH
export HF_HOME=/disk/wangzhe/.cache/huggingface
export HF_DATASETS_CACHE=/disk/wangzhe/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
MODEL_PATH=/disk/wangzhe/.cache/huggingface/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
```

如果你要固定单卡，可以额外执行：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## RTX 5090 说明

如果当前机器使用的是 RTX 5090，那么过旧的 PyTorch 版本会报类似错误：

```text
CUDA error: no kernel image is available for execution on the device
```

可行方向是使用较新的 CUDA 12.8 对应 PyTorch 版本。

在正式跑模型前，先验证 CUDA 基本计算是否正常：

```bash
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

## 推荐测试顺序

建议按照下面顺序执行：

1. CUDA sanity check
2. Baseline `sample.py`
3. Entropy-priority `sample.py`
4. Baseline `gsm8k --limit 20`
5. Entropy-priority `gsm8k --limit 20`
6. Optional `minerva_math --limit 20`

## 参数说明

下面解释测试命令中最关键的参数。理解这些参数后，你会更容易判断“当前是在测什么”。

### 通用参数

- `pretrained`
  - 含义：模型路径
  - 这里应当指向已经下载好的 LLaDA 模型目录
  - 必须是一个有效的本地目录，并且目录下包含 `config.json`

- `use_cache`
  - 含义：fastdLLM 的 cache 模式
  - 常见可选值：`none`、`prefix`、`dual`
  - 建议第一轮使用 `prefix`
  - 原因：它比 `none` 更贴近实际使用，同时一般比 `dual` 更稳

- `max_new_tokens`
  - 含义：最多生成多少个新 token
  - 值越大，推理时间和显存压力通常越大
  - 第一轮建议不要太大，例如 `128` 或 `256`

- `steps`
  - 含义：扩散/去噪的总步数
  - 通常和 `max_new_tokens` 先设置成相同，便于做稳定对照
  - 第一轮测试建议保持和已有示例一致，例如 `128` 或 `256`

- `block_size`
  - 含义：每个 block 中处理的 token 数
  - 它会影响 block 调度和缓存行为
  - 第一轮建议固定为 `32`

- `temperature`
  - 含义：采样温度
  - `0.0` 表示贪心式选择，更适合做稳定可复现对比
  - 第一轮建议使用 `0.0`

- `seed`
  - 含义：随机种子
  - 对比实验里必须固定
  - 推荐 `42`

- `--limit`
  - 含义：benchmark 中只评估前多少条样本
  - 只适合测试阶段，不适合报告正式结果
  - 第一轮建议 `20`

- `--num_fewshot`
  - 含义：few-shot 示例数
  - 应与原 benchmark 设定保持一致
  - `gsm8k` 常用 `5`
  - `minerva_math` 常用 `4`

### 你的新方法相关参数

- `entropy_min_tokens_per_step`
  - 含义：在启用 entropy-priority 时，每一步至少给多少个位置保留“高熵优先更新”的配额
  - `0` 表示完全关闭新方法，也就是基线
  - `1` 是最稳妥的第一组实验值

- `entropy_early_ratio`
  - 含义：只在前多少期步骤中启用 entropy-priority
  - 例如 `0.3` 表示只在前 30% 的有效步骤中应用高熵保留策略
  - 值太大可能让新策略影响过强，第一轮建议 `0.3`

- `entropy_top_k`
  - 含义：计算 entropy 时使用的 top-k 近似范围
  - `64` 或 `100` 是较合理的第一轮选择
  - 值更大通常更准确，但开销也可能更高

### 基线和新方法如何区分

基线：

- `entropy_min_tokens_per_step=0`

新方法：

- `entropy_min_tokens_per_step=1`
- `entropy_early_ratio=0.3`
- `entropy_top_k=64`

也就是说，最核心的对照变量是：

- 基线：不启用 entropy-priority
- 新方法：只改位置选择策略，不改模型、不改训练、不改总 transfer budget

## 1. Bas线冒烟测试

```bash
python -u /disk/wangzhe/dllm/examples/fastdllm/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --use_cache prefix \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --entropy_min_tokens_per_step 0
```

重点检查：

- script finishes successfully
- output text is normal
- no CUDA runtime error
- time and token speed look reasonable

## 2. 新方法冒烟测试

推荐第一组参数：

- `entropy_min_tokens_per_step=1`
- `entropy_early_ratio=0.3`
- `entropy_top_k=64`

Command:

```bash
python -u /disk/wangzhe/dllm/examples/fastdllm/llada/sample.py \
  --model_name_or_path "$MODEL_PATH" \
  --seed 42 \
  --visualize False \
  --use_cache prefix \
  --steps 128 \
  --max_new_tokens 128 \
  --block_size 32 \
  --temperature 0.0 \
  --entropy_min_tokens_per_step 1 \
  --entropy_early_ratio 0.3 \
  --entropy_top_k 64
```

和基线相比，主要观察：

- output difference
- speed difference
- whether generation becomes unstable

## 3. `gsm8k` 基线小样本测试

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"
```

## 4. `gsm8k` 新方法小样本测试

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[],entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64"
```

## 5. 可选的 `minerva_math` 小样本测试

Baseline:

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks minerva_math \
  --num_fewshot 4 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"
```

Entropy-priority:

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks minerva_math \
  --num_fewshot 4 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[],entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64"
```

## 日志记录

推荐先创建日志目录：

```bash
mkdir -p /disk/wangzhe/dllm/.logs
```

记录日志的示例：

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-baseline.log
```

Entropy-priority version:

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/fastdllm/llada/eval.py \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 20 \
  --model fastdllm_llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[],entropy_min_tokens_per_step=1,entropy_early_ratio=0.3,entropy_top_k=64" \
  2>&1 | tee /disk/wangzhe/dllm/.logs/gsm8k-entropy-priority.log
```

## 如何做公平对比

建议只做下面这种对比：

- baseline: `entropy_min_tokens_per_step=0`
- new method: `entropy_min_tokens_per_step>0`

下面这些条件必须固定：

- same model
- same prompt/task
- same `use_cache`
- same `steps`
- same `max_new_tokens`
- same `block_size`
- same seed

推荐第一轮对比方式：

- baseline: `use_cache=prefix`
- new method: `use_cache=prefix, entropy_min_tokens_per_step=1, entropy_early_ratio=0.3, entropy_top_k=64`

## 不建议一开始就做的事

- `humaneval_instruct_llada`
- `mbpp_instruct_llada`
- large-scale evaluation without `--limit`
- multi-GPU runs
- multiple concurrent runs on the same shared machine

## 常见报错与排查

### `pretrained=''`

原因：

- `MODEL_PATH` was not set

解决：

```bash
echo "$MODEL_PATH"
```

### 路径中包含换行

原因：

- model path was manually split into multiple lines

解决：

- use `MODEL_PATH=...`
- keep the entire `pretrained=$MODEL_PATH,...` string on one line

### `CUDA error: no kernel image is available for execution on the device`

原因：

- PyTorch build does not support the GPU architecture

解决：

- install a newer PyTorch build compatible with the current GPU

### `TRANSFORMERS_CACHE is deprecated`

原因：

- warning only

解决：

- prefer `HF_HOME`

## 建议的第一轮正式测试

在冒烟测试通过后，建议按下面顺序推进：

1. baseline `gsm8k --limit 20`
2. entropy-priority `gsm8k --limit 20`
3. compare correctness and speed
4. if reasonable, increase to `--limit 50`
5. only after that consider `minerva_math`
