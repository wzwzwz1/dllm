# Entropy-Priority Inference Change

## Summary

This change implements an inference-only scheduling update for dLLM samplers.

The goal is to avoid a pure confidence-first decoding order that keeps delaying
high-entropy fork tokens until the end of the denoising process. Instead, the
sampler can now reserve a small number of update slots for high-entropy masked
positions during the early phase of each block.

Model weights, training code, and model forward logic are unchanged. Only
sampler-side inference scheduling is modified.

## Files Changed

- `/Users/wz/code/dllm/dllm/core/samplers/utils.py`
- `/Users/wz/code/dllm/dllm/core/samplers/mdlm.py`
- `/Users/wz/code/dllm/dllm/pipelines/fastdllm/llada/sampler.py`
- `/Users/wz/code/dllm/scripts/tests/test_sampling_utils.py`

## What Changed

### 1. New entropy-aware sampler utilities

Added two inference helpers in `/Users/wz/code/dllm/dllm/core/samplers/utils.py`:

- `get_token_entropy(...)`
  Computes per-position entropy from logits.
  Supports optional top-k entropy approximation for lower overhead.

- `select_transfer_positions(...)`
  Selects masked positions to update this step.
  By default it behaves like confidence-first top-k selection.
  When entropy guidance is enabled, it:
  - reserves up to `entropy_min_tokens_per_step` slots for the highest-entropy masked positions
  - fills the remaining budget with the highest-confidence masked positions

### 2. MDLM sampler now supports entropy-priority scheduling

Updated `/Users/wz/code/dllm/dllm/core/samplers/mdlm.py`.

New config fields in `MDLMSamplerConfig`:

- `entropy_min_tokens_per_step: int = 0`
- `entropy_early_ratio: float = 0.3`
- `entropy_top_k: int | None = None`

Behavior:

- Default value `entropy_min_tokens_per_step=0` keeps old behavior unchanged.
- If `entropy_min_tokens_per_step > 0`, then during the early phase of each
  block (`ceil(effective_steps * entropy_early_ratio)` steps), the sampler:
  - computes entropy from current logits
  - reserves a small number of transfer slots for high-entropy masked positions
  - uses confidence for the rest of the transfer budget

This is applied to both:

- `sample(...)`
- `infill(...)`

### 3. Fast-dLLM LLaDA sampler now supports the same policy

Updated `/Users/wz/code/dllm/dllm/pipelines/fastdllm/llada/sampler.py`.

New config fields in `FastdLLMLLaDASamplerConfig`:

- `entropy_min_tokens_per_step: int = 0`
- `entropy_early_ratio: float = 0.3`
- `entropy_top_k: int | None = None`

The internal `_get_transfer_index(...)` logic now accepts entropy guidance and
applies it across:

- no-cache mode
- prefix-cache mode
- dual-cache mode

The update keeps the original transfer-count budget. The sampler still decides
how many positions to update each step using the existing schedule, but now it
can reserve part of that fixed budget for high-entropy positions.

## Before vs After

### Before

Each step was effectively confidence-first:

- compute logits
- predict tokens
- score positions by confidence
- update the top-confidence masked positions

This tends to postpone uncertain but structurally important positions.

### After

Each early step becomes mixed scheduling:

- compute logits
- predict tokens
- compute confidence
- optionally compute entropy
- reserve a few slots for the highest-entropy masked positions
- fill the remaining slots with the highest-confidence masked positions

So the sampler still converges efficiently, but it no longer lets all difficult
fork positions drift to the very end.

## How To Use

If you want the old behavior, do nothing.

If you want entropy-priority scheduling, enable it in sampler config.

Example for `MDLMSamplerConfig`:

```python
sampler_config.entropy_min_tokens_per_step = 1
sampler_config.entropy_early_ratio = 0.3
sampler_config.entropy_top_k = 64
```

Example for `FastdLLMLLaDASamplerConfig`:

```python
sampler_config.entropy_min_tokens_per_step = 1
sampler_config.entropy_early_ratio = 0.3
sampler_config.entropy_top_k = 64
```

Recommended first try:

- `entropy_min_tokens_per_step = 1`
- `entropy_early_ratio = 0.3`
- `entropy_top_k = 64` or `100`

## Scope And Limits

- This is an inference-only change.
- It does not modify model weights or training.
- It does not yet add runtime logging for fork-token entropy diagnostics.
- It does not yet implement soft-token / EvoToken-lite behavior.
- It does not change the transfer-count schedule itself, only which positions
  consume that budget.

## Verification

What was checked:

- Python syntax compilation succeeded for the modified sampler and test files.
- Unit tests for the new utility functions were added.

What could not be fully run in the current environment:

- `pytest /Users/wz/code/dllm/scripts/tests/test_sampling_utils.py -q`

This failed during collection because the active Python environment does not
have `torch` installed.
