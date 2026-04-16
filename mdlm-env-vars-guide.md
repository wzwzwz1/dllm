# MDLM 环境变量说明

本文档整理 `/disk/wangzhe/dllm` 里运行单元测试、`sample.py` 冒烟测试、以及 `gsm8k_cot --limit 20` 评测时常用的环境变量。

当前模型路径固定为：

```bash
/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
```

## 1. 最小必需环境变量

如果你只是运行本仓库的 Python 测试或评测脚本，最常用的是下面这些：

```bash
export PYTHONPATH=.:$PYTHONPATH
export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export HF_HOME=/disk/wangzhe/.cache/huggingface
export HF_DATASETS_CACHE=/disk/wangzhe/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
```

含义如下：

- `PYTHONPATH`
  - 让当前仓库代码能被直接导入
- `MODEL_PATH`
  - 指向本地 LLaDA 模型目录
- `HF_HOME`
  - Hugging Face 主缓存目录
- `HF_DATASETS_CACHE`
  - datasets 缓存目录
- `HF_ALLOW_CODE_EVAL`
  - 允许部分评测任务执行代码相关评估逻辑
- `HF_DATASETS_TRUST_REMOTE_CODE`
  - 允许 datasets 使用远端自定义脚本

## 2. 推荐完整初始化命令

每次新开 shell 后，建议按下面顺序执行：

```bash
source ~/.zshrc
source /home/wangzhe/miniconda3/etc/profile.d/conda.sh
conda activate dllm
cd /disk/wangzhe/dllm

export PYTHONPATH=.:$PYTHONPATH
export MODEL_PATH=/disk/wangzhe/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export HF_HOME=/disk/wangzhe/.cache/huggingface
export HF_DATASETS_CACHE=/disk/wangzhe/.cache/huggingface/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
```

## 3. 如果你要固定单卡

```bash
export CUDA_VISIBLE_DEVICES=0
```

如果你在共享机器上跑评测，这个变量很有用。

## 4. 如果你想减少旧变量干扰

你之前看到过：

```text
Using `TRANSFORMERS_CACHE` is deprecated
```

这不是致命错误，但如果你想减少这类提示，建议优先使用 `HF_HOME`，不要再额外设置：

```bash
unset TRANSFORMERS_CACHE
```

## 5. 运行前自检

建议在真正跑评测前检查这几个变量：

```bash
echo "$PYTHONPATH"
echo "$MODEL_PATH"
echo "$HF_HOME"
echo "$HF_DATASETS_CACHE"
```

尤其要确认：

```bash
echo "$MODEL_PATH"
```

输出必须是完整的一整行路径，不能是空字符串。  
如果这里为空，后面的 `eval.py` 会把参数解析成：

```text
pretrained=
```

然后直接失败。

## 6. 最常用的两条命令

### 6.1 单元测试

```bash
pytest /disk/wangzhe/dllm/scripts/tests/test_sampling_utils.py -v
```

### 6.2 `gsm8k_cot --limit 20`

```bash
accelerate launch --num_processes 1 /disk/wangzhe/dllm/dllm/pipelines/llada/eval.py \
  --tasks gsm8k_cot \
  --num_fewshot 5 \
  --limit 20 \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL_PATH,max_new_tokens=512,steps=512,block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348],entropy_min_tokens_per_step=0"
```

## 7. 建议

- 新开一个 shell 就重新执行一次完整初始化命令
- 每次跑 `eval.py` 前先 `echo "$MODEL_PATH"`
- 优先使用 `HF_HOME`，不要混用太多旧 Hugging Face 环境变量
- 如果你切换模型，只改 `MODEL_PATH` 即可，其余变量通常不需要动
