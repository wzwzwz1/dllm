"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import (
    add_gumbel_noise,
    build_priority_scores,
    build_structure_prior_scores,
    compute_entropy_phase_scale,
    compute_entropy_trigger_counts,
    compute_tentative_finalize_mask,
    compute_tentative_rollback_mask,
    get_num_transfer_tokens,
    get_top1_margin,
    get_token_entropy,
    select_entropy_candidate_mask,
    select_transfer_positions,
    update_entropy_credit,
    update_tentative_stats,
)


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    enable_entropy_priority: bool = False
    enable_tentative_commit: bool = False
    enable_targeted_remask: bool = False
    enable_structure_priority: bool = False
    enable_priority_age_bonus: bool = False
    enable_sampler_diagnostics: bool = False
    enable_entropy_credit_scheduler: bool = True
    entropy_min_tokens_per_step: int = 0
    entropy_early_ratio: float = 0.3
    entropy_top_k: int | None = None
    tentative_budget_ratio: float = 0.1
    entropy_credit_rate: float = 0.35
    entropy_warmup_ratio: float = 0.05
    entropy_active_end_ratio: float = 0.20
    entropy_end_ratio: float = 0.30
    entropy_max_trigger_per_step: int = 1
    entropy_use_quality_gate: bool = True
    entropy_conf_floor: float = 0.15
    entropy_age_threshold: int = 2
    entropy_top_candidate_pool: int = 4
    tentative_min_hold_steps: int = 1
    tentative_stable_steps: int = 2
    tentative_max_hold_steps: int = 3
    tentative_final_prob_thresh: float = 0.82
    tentative_final_margin_thresh: float = 0.35
    remask_rollback_prob_thresh: float = 0.45
    remask_flip_thresh: int = 2
    priority_entropy_weight: float = 1.0
    priority_structure_weight: float = 0.8
    priority_age_weight: float = 0.4
    priority_confidence_weight: float = 0.6
    structure_prior_mode: str = "none"
    structure_prior_strength: float = 1.0
    diagnostic_log_interval: int = 1
    diagnostic_collect_token_events: bool = False


def _init_sampler_diagnostics(*, enabled: bool, collect_token_events: bool) -> dict:
    diagnostics = {
        "tentative_enter_count": 0,
        "tentative_finalize_count": 0,
        "tentative_rollback_count": 0,
        "baseline_finalize_count": 0,
        "entropy_trigger_count": 0,
        "quota_conf_total": 0,
        "quota_tent_total": 0,
    }
    if enabled and collect_token_events:
        diagnostics["token_event_log"] = []
    return diagnostics


def _append_token_events(
    diagnostics: dict,
    *,
    event_name: str,
    event_mask: torch.Tensor,
    step_idx: int,
):
    if "token_event_log" not in diagnostics or not event_mask.any():
        return
    for batch_idx, pos_idx in event_mask.nonzero(as_tuple=False).tolist():
        diagnostics["token_event_log"].append(
            {"event": event_name, "step": step_idx, "batch": batch_idx, "pos": pos_idx}
        )


def _clear_positions(
    clear_mask: torch.Tensor,
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.where(clear_mask, torch.zeros_like(tensor), tensor) for tensor in tensors
    )


def _compute_entropy_trigger_plan(
    *,
    tokenizer,
    x: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    candidate_mask: torch.Tensor,
    deferred_age: torch.Tensor,
    total_budget: torch.Tensor,
    entropy_credit: torch.Tensor,
    step_ratio: float,
    enable_entropy_priority: bool,
    enable_entropy_credit_scheduler: bool,
    enable_structure_priority: bool,
    enable_priority_age_bonus: bool,
    entropy_top_k: int | None,
    entropy_credit_rate: float,
    entropy_warmup_ratio: float,
    entropy_active_end_ratio: float,
    entropy_end_ratio: float,
    entropy_max_trigger_per_step: int,
    entropy_use_quality_gate: bool,
    entropy_conf_floor: float,
    entropy_age_threshold: int,
    entropy_top_candidate_pool: int,
    priority_entropy_weight: float,
    priority_structure_weight: float,
    priority_age_weight: float,
    priority_confidence_weight: float,
    structure_prior_mode: str,
    structure_prior_strength: float,
    logits: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_counts = torch.zeros_like(total_budget, dtype=torch.long)
    zero_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if not enable_entropy_priority or not enable_entropy_credit_scheduler:
        return None, entropy_credit, zero_counts, zero_mask

    phase_scale = compute_entropy_phase_scale(
        step_ratio,
        entropy_warmup_ratio,
        entropy_active_end_ratio,
        entropy_end_ratio,
    )
    if phase_scale <= 0:
        return None, entropy_credit, zero_counts, zero_mask

    entropy_scores = get_token_entropy(logits, top_k=entropy_top_k)
    entropy_credit = update_entropy_credit(
        entropy_credit,
        phase_scale=phase_scale,
        credit_rate=entropy_credit_rate,
    )

    structure_scores = build_structure_prior_scores(
        tokenizer,
        x0,
        candidate_mask,
        structure_prior_mode if enable_structure_priority else "none",
        structure_prior_strength,
        context_tokens=x,
    )
    age_scores = (
        deferred_age.to(torch.float32)
        if enable_priority_age_bonus
        else torch.zeros_like(confidence, dtype=torch.float32)
    )
    priority_scores = build_priority_scores(
        entropy_scores,
        confidence=confidence,
        structure_scores=structure_scores,
        age_scores=age_scores,
        entropy_weight=priority_entropy_weight,
        structure_weight=priority_structure_weight,
        age_weight=priority_age_weight,
        confidence_weight=priority_confidence_weight,
    )
    entropy_candidate_mask = select_entropy_candidate_mask(
        priority_scores=priority_scores,
        candidate_mask=candidate_mask,
        structure_scores=structure_scores,
        age_scores=age_scores,
        confidence=confidence,
        top_candidate_pool=entropy_top_candidate_pool,
        use_quality_gate=entropy_use_quality_gate,
        confidence_floor=entropy_conf_floor,
        age_threshold=entropy_age_threshold,
    )
    trigger_counts = compute_entropy_trigger_counts(
        entropy_credit,
        candidate_exists=entropy_candidate_mask.any(dim=1),
        max_trigger_per_step=entropy_max_trigger_per_step,
    )
    trigger_counts = torch.minimum(trigger_counts, total_budget.to(torch.long))
    entropy_candidate_mask = entropy_candidate_mask & trigger_counts.unsqueeze(1).bool()
    entropy_credit = entropy_credit - trigger_counts.to(entropy_credit.dtype)
    return entropy_scores, entropy_credit, trigger_counts, entropy_candidate_mask


def _apply_extended_commit_strategy(
    *,
    tokenizer,
    x: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    margin_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    conf_target_counts: torch.Tensor,
    tentative_target_counts: torch.Tensor,
    tentative_candidate_mask: torch.Tensor,
    step_idx: int,
    mask_id: int,
    final_mask: torch.Tensor,
    tentative_mask: torch.Tensor,
    deferred_age: torch.Tensor,
    tentative_token_ids: torch.Tensor,
    tentative_age: torch.Tensor,
    tentative_flip_count: torch.Tensor,
    tentative_last_top1: torch.Tensor,
    tentative_stable_run: torch.Tensor,
    tentative_last_conf: torch.Tensor,
    tentative_last_margin: torch.Tensor,
    enable_tentative_commit: bool,
    enable_targeted_remask: bool,
    tentative_min_hold_steps: int,
    tentative_stable_steps: int,
    tentative_max_hold_steps: int,
    tentative_final_prob_thresh: float,
    tentative_final_margin_thresh: float,
    remask_rollback_prob_thresh: float,
    remask_flip_thresh: int,
    diagnostics: dict,
) -> tuple[torch.Tensor, ...]:
    released_this_step = torch.zeros_like(candidate_mask)
    if tentative_mask.any():
        (
            tentative_token_ids,
            tentative_age,
            tentative_flip_count,
            tentative_last_top1,
            tentative_stable_run,
            tentative_last_conf,
            tentative_last_margin,
        ) = update_tentative_stats(
            current_top1=x0,
            current_conf=confidence,
            current_margin=margin_scores,
            tentative_mask=tentative_mask,
            tentative_token_ids=tentative_token_ids,
            tentative_age=tentative_age,
            tentative_flip_count=tentative_flip_count,
            tentative_last_top1=tentative_last_top1,
            tentative_stable_run=tentative_stable_run,
            tentative_last_conf=tentative_last_conf,
            tentative_last_margin=tentative_last_margin,
        )

        finalize_mask = compute_tentative_finalize_mask(
            tentative_mask=tentative_mask,
            tentative_age=tentative_age,
            tentative_stable_run=tentative_stable_run,
            tentative_last_conf=tentative_last_conf,
            tentative_last_margin=tentative_last_margin,
            min_hold_steps=tentative_min_hold_steps,
            stable_steps=tentative_stable_steps,
            final_prob_thresh=tentative_final_prob_thresh,
            final_margin_thresh=tentative_final_margin_thresh,
        )
        rollback_mask = (
            compute_tentative_rollback_mask(
                tentative_mask=tentative_mask,
                tentative_age=tentative_age,
                tentative_flip_count=tentative_flip_count,
                tentative_last_conf=tentative_last_conf,
                tentative_stable_run=tentative_stable_run,
                max_hold_steps=tentative_max_hold_steps,
                rollback_prob_thresh=remask_rollback_prob_thresh,
                flip_thresh=remask_flip_thresh,
                stable_steps=tentative_stable_steps,
            )
            if enable_targeted_remask
            else torch.zeros_like(tentative_mask)
        )
        keep_mask = tentative_mask & ~finalize_mask & ~rollback_mask

        x[keep_mask] = tentative_token_ids[keep_mask]
        x[finalize_mask] = tentative_token_ids[finalize_mask]
        x[rollback_mask] = mask_id

        final_mask = final_mask | finalize_mask
        tentative_mask = keep_mask
        deferred_age = torch.where(
            finalize_mask | rollback_mask, torch.zeros_like(deferred_age), deferred_age
        )

        (
            tentative_token_ids,
            tentative_age,
            tentative_flip_count,
            tentative_last_top1,
            tentative_stable_run,
            tentative_last_conf,
            tentative_last_margin,
        ) = _clear_positions(
            finalize_mask | rollback_mask,
            tentative_token_ids,
            tentative_age,
            tentative_flip_count,
            tentative_last_top1,
            tentative_stable_run,
            tentative_last_conf,
            tentative_last_margin,
        )

        diagnostics["tentative_finalize_count"] += int(finalize_mask.sum().item())
        diagnostics["tentative_rollback_count"] += int(rollback_mask.sum().item())
        released_this_step = finalize_mask | rollback_mask
        _append_token_events(
            diagnostics, event_name="tentative_finalize", event_mask=finalize_mask, step_idx=step_idx
        )
        _append_token_events(
            diagnostics, event_name="tentative_rollback", event_mask=rollback_mask, step_idx=step_idx
        )

    remaining_mask = (
        candidate_mask
        & (x == mask_id)
        & ~tentative_mask
        & ~final_mask
        & ~released_this_step
    )

    reserved_tentative = tentative_candidate_mask & remaining_mask
    conf_mask = remaining_mask & ~reserved_tentative
    conf_transfer = select_transfer_positions(
        confidence=confidence,
        mask_index=conf_mask,
        target_counts=conf_target_counts.to(torch.long),
    )
    x[conf_transfer] = x0[conf_transfer]
    final_mask = final_mask | conf_transfer
    diagnostics["baseline_finalize_count"] += int(conf_transfer.sum().item())
    diagnostics["quota_conf_total"] += int(conf_target_counts.sum().item())
    _append_token_events(
        diagnostics, event_name="baseline_finalize", event_mask=conf_transfer, step_idx=step_idx
    )

    remaining_mask = remaining_mask & ~conf_transfer

    if enable_tentative_commit and tentative_target_counts.any():
        tentative_transfer = reserved_tentative & remaining_mask
        x[tentative_transfer] = x0[tentative_transfer]
        tentative_mask = tentative_mask | tentative_transfer
        tentative_token_ids = torch.where(
            tentative_transfer, x0, tentative_token_ids
        )
        tentative_age = torch.where(
            tentative_transfer, torch.zeros_like(tentative_age), tentative_age
        )
        tentative_flip_count = torch.where(
            tentative_transfer,
            torch.zeros_like(tentative_flip_count),
            tentative_flip_count,
        )
        tentative_last_top1 = torch.where(
            tentative_transfer, x0, tentative_last_top1
        )
        tentative_stable_run = torch.where(
            tentative_transfer,
            torch.ones_like(tentative_stable_run),
            tentative_stable_run,
        )
        tentative_last_conf = torch.where(
            tentative_transfer, confidence, tentative_last_conf
        )
        tentative_last_margin = torch.where(
            tentative_transfer, margin_scores, tentative_last_margin
        )
        diagnostics["tentative_enter_count"] += int(tentative_transfer.sum().item())
        diagnostics["quota_tent_total"] += int(tentative_target_counts.sum().item())
        _append_token_events(
            diagnostics,
            event_name="tentative_enter",
            event_mask=tentative_transfer,
            step_idx=step_idx,
        )

    unresolved_mask = candidate_mask & (x == mask_id)
    deferred_age = torch.where(
        unresolved_mask,
        deferred_age + 1,
        torch.zeros_like(deferred_age),
    )

    return (
        x,
        final_mask,
        tentative_mask,
        deferred_age,
        tentative_token_ids,
        tentative_age,
        tentative_flip_count,
        tentative_last_top1,
        tentative_stable_run,
        tentative_last_conf,
        tentative_last_margin,
    )


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        requested_steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        entropy_min_tokens_per_step = kwargs.get(
            "entropy_min_tokens_per_step", config.entropy_min_tokens_per_step
        )
        entropy_early_ratio = kwargs.get(
            "entropy_early_ratio", config.entropy_early_ratio
        )
        entropy_top_k = kwargs.get("entropy_top_k", config.entropy_top_k)
        enable_entropy_credit_scheduler = kwargs.get(
            "enable_entropy_credit_scheduler", config.enable_entropy_credit_scheduler
        )
        entropy_credit_rate = kwargs.get(
            "entropy_credit_rate", config.entropy_credit_rate
        )
        entropy_warmup_ratio = kwargs.get(
            "entropy_warmup_ratio", config.entropy_warmup_ratio
        )
        entropy_active_end_ratio = kwargs.get(
            "entropy_active_end_ratio", config.entropy_active_end_ratio
        )
        entropy_end_ratio = kwargs.get(
            "entropy_end_ratio", config.entropy_end_ratio
        )
        entropy_max_trigger_per_step = kwargs.get(
            "entropy_max_trigger_per_step", config.entropy_max_trigger_per_step
        )
        entropy_use_quality_gate = kwargs.get(
            "entropy_use_quality_gate", config.entropy_use_quality_gate
        )
        entropy_conf_floor = kwargs.get(
            "entropy_conf_floor", config.entropy_conf_floor
        )
        entropy_age_threshold = kwargs.get(
            "entropy_age_threshold", config.entropy_age_threshold
        )
        entropy_top_candidate_pool = kwargs.get(
            "entropy_top_candidate_pool", config.entropy_top_candidate_pool
        )
        enable_entropy_priority = kwargs.get(
            "enable_entropy_priority", config.enable_entropy_priority
        )
        enable_tentative_commit = kwargs.get(
            "enable_tentative_commit", config.enable_tentative_commit
        )
        enable_targeted_remask = kwargs.get(
            "enable_targeted_remask", config.enable_targeted_remask
        )
        enable_structure_priority = kwargs.get(
            "enable_structure_priority", config.enable_structure_priority
        )
        enable_priority_age_bonus = kwargs.get(
            "enable_priority_age_bonus", config.enable_priority_age_bonus
        )
        enable_sampler_diagnostics = kwargs.get(
            "enable_sampler_diagnostics", config.enable_sampler_diagnostics
        )
        tentative_budget_ratio = kwargs.get(
            "tentative_budget_ratio", config.tentative_budget_ratio
        )
        tentative_min_hold_steps = kwargs.get(
            "tentative_min_hold_steps", config.tentative_min_hold_steps
        )
        tentative_stable_steps = kwargs.get(
            "tentative_stable_steps", config.tentative_stable_steps
        )
        tentative_max_hold_steps = kwargs.get(
            "tentative_max_hold_steps", config.tentative_max_hold_steps
        )
        tentative_final_prob_thresh = kwargs.get(
            "tentative_final_prob_thresh", config.tentative_final_prob_thresh
        )
        tentative_final_margin_thresh = kwargs.get(
            "tentative_final_margin_thresh", config.tentative_final_margin_thresh
        )
        remask_rollback_prob_thresh = kwargs.get(
            "remask_rollback_prob_thresh", config.remask_rollback_prob_thresh
        )
        remask_flip_thresh = kwargs.get(
            "remask_flip_thresh", config.remask_flip_thresh
        )
        priority_entropy_weight = kwargs.get(
            "priority_entropy_weight", config.priority_entropy_weight
        )
        priority_structure_weight = kwargs.get(
            "priority_structure_weight", config.priority_structure_weight
        )
        priority_age_weight = kwargs.get(
            "priority_age_weight", config.priority_age_weight
        )
        priority_confidence_weight = kwargs.get(
            "priority_confidence_weight", config.priority_confidence_weight
        )
        structure_prior_mode = kwargs.get(
            "structure_prior_mode", config.structure_prior_mode
        )
        structure_prior_strength = kwargs.get(
            "structure_prior_strength", config.structure_prior_strength
        )
        diagnostic_collect_token_events = kwargs.get(
            "diagnostic_collect_token_events", config.diagnostic_collect_token_events
        )

        assert 1 <= block_size
        assert 1 <= requested_steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(requested_steps / num_blocks)
        total_planned_steps = steps_per_block * num_blocks
        histories = [x.clone()] if return_dict else None
        diagnostics = _init_sampler_diagnostics(
            enabled=enable_sampler_diagnostics,
            collect_token_events=diagnostic_collect_token_events,
        )
        entropy_credit = torch.zeros((B,), device=x.device, dtype=torch.float32)
        global_step_idx = 0
        use_tentative_features = enable_tentative_commit
        if use_tentative_features:
            final_mask = (x != mask_id) & attention_mask.bool()
            tentative_mask = torch.zeros_like(final_mask)
            deferred_age = torch.zeros_like(x, dtype=torch.long)
            tentative_token_ids = torch.zeros_like(x, dtype=torch.long)
            tentative_age = torch.zeros_like(x, dtype=torch.long)
            tentative_flip_count = torch.zeros_like(x, dtype=torch.long)
            tentative_last_top1 = torch.full_like(x, fill_value=-1, dtype=torch.long)
            tentative_stable_run = torch.zeros_like(x, dtype=torch.long)
            tentative_last_conf = torch.zeros_like(x, dtype=torch.float32)
            tentative_last_margin = torch.zeros_like(x, dtype=torch.float32)

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                candidate_mask = mask_index.clone()
                for j in range(B):
                    block_start = prompt_lens[j] + b * block_size
                    block_end = prompt_lens[j] + (b + 1) * block_size
                    candidate_mask[j, :block_start] = False
                    candidate_mask[j, block_end:] = False

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(candidate_mask, x0_p, -np.inf)
                margin_scores = torch.where(
                    candidate_mask,
                    get_top1_margin(logits),
                    torch.zeros_like(confidence, dtype=torch.float32),
                )

                step_ratio = (
                    global_step_idx / total_planned_steps
                    if total_planned_steps > 0
                    else 0.0
                )
                (
                    entropy_scores,
                    entropy_credit,
                    entropy_trigger_counts,
                    entropy_candidate_mask,
                ) = _compute_entropy_trigger_plan(
                    tokenizer=self.tokenizer,
                    x=x,
                    x0=x0,
                    confidence=confidence,
                    candidate_mask=candidate_mask,
                    deferred_age=deferred_age if use_tentative_features else torch.zeros_like(x, dtype=torch.long),
                    total_budget=num_transfer_tokens[:, i],
                    entropy_credit=entropy_credit,
                    step_ratio=step_ratio,
                    enable_entropy_priority=enable_entropy_priority,
                    enable_entropy_credit_scheduler=enable_entropy_credit_scheduler,
                    enable_structure_priority=enable_structure_priority,
                    enable_priority_age_bonus=enable_priority_age_bonus,
                    entropy_top_k=entropy_top_k,
                    entropy_credit_rate=entropy_credit_rate,
                    entropy_warmup_ratio=entropy_warmup_ratio,
                    entropy_active_end_ratio=entropy_active_end_ratio,
                    entropy_end_ratio=entropy_end_ratio,
                    entropy_max_trigger_per_step=entropy_max_trigger_per_step,
                    entropy_use_quality_gate=entropy_use_quality_gate,
                    entropy_conf_floor=entropy_conf_floor,
                    entropy_age_threshold=entropy_age_threshold,
                    entropy_top_candidate_pool=entropy_top_candidate_pool,
                    priority_entropy_weight=priority_entropy_weight,
                    priority_structure_weight=priority_structure_weight,
                    priority_age_weight=priority_age_weight,
                    priority_confidence_weight=priority_confidence_weight,
                    structure_prior_mode=structure_prior_mode,
                    structure_prior_strength=structure_prior_strength,
                    logits=logits,
                )
                diagnostics["entropy_trigger_count"] += int(
                    entropy_trigger_counts.sum().item()
                )

                if use_tentative_features:
                    conf_target_counts = torch.clamp(
                        num_transfer_tokens[:, i].to(torch.long) - entropy_trigger_counts,
                        min=0,
                    )
                    tentative_target_counts = (
                        entropy_trigger_counts
                        if enable_tentative_commit
                        else torch.zeros_like(entropy_trigger_counts)
                    )
                    (
                        x,
                        final_mask,
                        tentative_mask,
                        deferred_age,
                        tentative_token_ids,
                        tentative_age,
                        tentative_flip_count,
                        tentative_last_top1,
                        tentative_stable_run,
                        tentative_last_conf,
                        tentative_last_margin,
                    ) = _apply_extended_commit_strategy(
                        tokenizer=self.tokenizer,
                        x=x,
                        x0=x0,
                        confidence=confidence,
                        margin_scores=margin_scores,
                        candidate_mask=candidate_mask,
                        conf_target_counts=conf_target_counts,
                        tentative_target_counts=tentative_target_counts,
                        tentative_candidate_mask=entropy_candidate_mask,
                        step_idx=i,
                        mask_id=mask_id,
                        final_mask=final_mask,
                        tentative_mask=tentative_mask,
                        deferred_age=deferred_age,
                        tentative_token_ids=tentative_token_ids,
                        tentative_age=tentative_age,
                        tentative_flip_count=tentative_flip_count,
                        tentative_last_top1=tentative_last_top1,
                        tentative_stable_run=tentative_stable_run,
                        tentative_last_conf=tentative_last_conf,
                        tentative_last_margin=tentative_last_margin,
                        enable_tentative_commit=enable_tentative_commit,
                        enable_targeted_remask=enable_targeted_remask,
                        tentative_min_hold_steps=tentative_min_hold_steps,
                        tentative_stable_steps=tentative_stable_steps,
                        tentative_max_hold_steps=tentative_max_hold_steps,
                        tentative_final_prob_thresh=tentative_final_prob_thresh,
                        tentative_final_margin_thresh=tentative_final_margin_thresh,
                        remask_rollback_prob_thresh=remask_rollback_prob_thresh,
                        remask_flip_thresh=remask_flip_thresh,
                        diagnostics=diagnostics,
                    )
                else:
                    entropy_transfer = entropy_candidate_mask
                    x[entropy_transfer] = x0[entropy_transfer]
                    confidence_mask = candidate_mask & ~entropy_transfer
                    confidence_counts = torch.clamp(
                        num_transfer_tokens[:, i].to(torch.long) - entropy_trigger_counts,
                        min=0,
                    )
                    transfer_index = select_transfer_positions(
                        confidence=confidence,
                        mask_index=confidence_mask,
                        target_counts=confidence_counts,
                    )
                    x[transfer_index] = x0[transfer_index]
                    if enable_sampler_diagnostics:
                        diagnostics["baseline_finalize_count"] += int(
                            (transfer_index | entropy_transfer).sum().item()
                        )
                        diagnostics["quota_conf_total"] += int(
                            confidence_counts.sum().item()
                        )
                global_step_idx += 1
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        self._last_sampler_diagnostics = diagnostics
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        requested_steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        entropy_min_tokens_per_step = kwargs.get(
            "entropy_min_tokens_per_step", config.entropy_min_tokens_per_step
        )
        entropy_early_ratio = kwargs.get(
            "entropy_early_ratio", config.entropy_early_ratio
        )
        entropy_top_k = kwargs.get("entropy_top_k", config.entropy_top_k)
        enable_entropy_credit_scheduler = kwargs.get(
            "enable_entropy_credit_scheduler", config.enable_entropy_credit_scheduler
        )
        entropy_credit_rate = kwargs.get(
            "entropy_credit_rate", config.entropy_credit_rate
        )
        entropy_warmup_ratio = kwargs.get(
            "entropy_warmup_ratio", config.entropy_warmup_ratio
        )
        entropy_active_end_ratio = kwargs.get(
            "entropy_active_end_ratio", config.entropy_active_end_ratio
        )
        entropy_end_ratio = kwargs.get(
            "entropy_end_ratio", config.entropy_end_ratio
        )
        entropy_max_trigger_per_step = kwargs.get(
            "entropy_max_trigger_per_step", config.entropy_max_trigger_per_step
        )
        entropy_use_quality_gate = kwargs.get(
            "entropy_use_quality_gate", config.entropy_use_quality_gate
        )
        entropy_conf_floor = kwargs.get(
            "entropy_conf_floor", config.entropy_conf_floor
        )
        entropy_age_threshold = kwargs.get(
            "entropy_age_threshold", config.entropy_age_threshold
        )
        entropy_top_candidate_pool = kwargs.get(
            "entropy_top_candidate_pool", config.entropy_top_candidate_pool
        )
        enable_entropy_priority = kwargs.get(
            "enable_entropy_priority", config.enable_entropy_priority
        )
        enable_tentative_commit = kwargs.get(
            "enable_tentative_commit", config.enable_tentative_commit
        )
        enable_targeted_remask = kwargs.get(
            "enable_targeted_remask", config.enable_targeted_remask
        )
        enable_structure_priority = kwargs.get(
            "enable_structure_priority", config.enable_structure_priority
        )
        enable_priority_age_bonus = kwargs.get(
            "enable_priority_age_bonus", config.enable_priority_age_bonus
        )
        enable_sampler_diagnostics = kwargs.get(
            "enable_sampler_diagnostics", config.enable_sampler_diagnostics
        )
        tentative_budget_ratio = kwargs.get(
            "tentative_budget_ratio", config.tentative_budget_ratio
        )
        tentative_min_hold_steps = kwargs.get(
            "tentative_min_hold_steps", config.tentative_min_hold_steps
        )
        tentative_stable_steps = kwargs.get(
            "tentative_stable_steps", config.tentative_stable_steps
        )
        tentative_max_hold_steps = kwargs.get(
            "tentative_max_hold_steps", config.tentative_max_hold_steps
        )
        tentative_final_prob_thresh = kwargs.get(
            "tentative_final_prob_thresh", config.tentative_final_prob_thresh
        )
        tentative_final_margin_thresh = kwargs.get(
            "tentative_final_margin_thresh", config.tentative_final_margin_thresh
        )
        remask_rollback_prob_thresh = kwargs.get(
            "remask_rollback_prob_thresh", config.remask_rollback_prob_thresh
        )
        remask_flip_thresh = kwargs.get(
            "remask_flip_thresh", config.remask_flip_thresh
        )
        priority_entropy_weight = kwargs.get(
            "priority_entropy_weight", config.priority_entropy_weight
        )
        priority_structure_weight = kwargs.get(
            "priority_structure_weight", config.priority_structure_weight
        )
        priority_age_weight = kwargs.get(
            "priority_age_weight", config.priority_age_weight
        )
        priority_confidence_weight = kwargs.get(
            "priority_confidence_weight", config.priority_confidence_weight
        )
        structure_prior_mode = kwargs.get(
            "structure_prior_mode", config.structure_prior_mode
        )
        structure_prior_strength = kwargs.get(
            "structure_prior_strength", config.structure_prior_strength
        )
        diagnostic_collect_token_events = kwargs.get(
            "diagnostic_collect_token_events", config.diagnostic_collect_token_events
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= requested_steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(requested_steps / num_blocks)
        total_planned_steps = steps_per_block * num_blocks
        histories = [x.clone()] if return_dict else None
        diagnostics = _init_sampler_diagnostics(
            enabled=enable_sampler_diagnostics,
            collect_token_events=diagnostic_collect_token_events,
        )
        entropy_credit = torch.zeros((B,), device=x.device, dtype=torch.float32)
        global_step_idx = 0
        use_tentative_features = enable_tentative_commit
        if use_tentative_features:
            final_mask = (x != mask_id) & attention_mask.bool()
            tentative_mask = torch.zeros_like(final_mask)
            deferred_age = torch.zeros_like(x, dtype=torch.long)
            tentative_token_ids = torch.zeros_like(x, dtype=torch.long)
            tentative_age = torch.zeros_like(x, dtype=torch.long)
            tentative_flip_count = torch.zeros_like(x, dtype=torch.long)
            tentative_last_top1 = torch.full_like(x, fill_value=-1, dtype=torch.long)
            tentative_stable_run = torch.zeros_like(x, dtype=torch.long)
            tentative_last_conf = torch.zeros_like(x, dtype=torch.float32)
            tentative_last_margin = torch.zeros_like(x, dtype=torch.float32)

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                candidate_mask = mask_index_full.clone()
                for j in range(B):
                    end_j = start + widths[j]
                    candidate_mask[j, :start] = False
                    candidate_mask[j, end_j:] = False

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(candidate_mask, x0_p, -np.inf)
                margin_scores = torch.where(
                    candidate_mask,
                    get_top1_margin(logits),
                    torch.zeros_like(confidence, dtype=torch.float32),
                )

                step_ratio = (
                    global_step_idx / total_planned_steps
                    if total_planned_steps > 0
                    else 0.0
                )
                (
                    entropy_scores,
                    entropy_credit,
                    entropy_trigger_counts,
                    entropy_candidate_mask,
                ) = _compute_entropy_trigger_plan(
                    tokenizer=self.tokenizer,
                    x=x,
                    x0=x0,
                    confidence=confidence,
                    candidate_mask=candidate_mask,
                    deferred_age=deferred_age if use_tentative_features else torch.zeros_like(x, dtype=torch.long),
                    total_budget=num_transfer_tokens[:, s],
                    entropy_credit=entropy_credit,
                    step_ratio=step_ratio,
                    enable_entropy_priority=enable_entropy_priority,
                    enable_entropy_credit_scheduler=enable_entropy_credit_scheduler,
                    enable_structure_priority=enable_structure_priority,
                    enable_priority_age_bonus=enable_priority_age_bonus,
                    entropy_top_k=entropy_top_k,
                    entropy_credit_rate=entropy_credit_rate,
                    entropy_warmup_ratio=entropy_warmup_ratio,
                    entropy_active_end_ratio=entropy_active_end_ratio,
                    entropy_end_ratio=entropy_end_ratio,
                    entropy_max_trigger_per_step=entropy_max_trigger_per_step,
                    entropy_use_quality_gate=entropy_use_quality_gate,
                    entropy_conf_floor=entropy_conf_floor,
                    entropy_age_threshold=entropy_age_threshold,
                    entropy_top_candidate_pool=entropy_top_candidate_pool,
                    priority_entropy_weight=priority_entropy_weight,
                    priority_structure_weight=priority_structure_weight,
                    priority_age_weight=priority_age_weight,
                    priority_confidence_weight=priority_confidence_weight,
                    structure_prior_mode=structure_prior_mode,
                    structure_prior_strength=structure_prior_strength,
                    logits=logits,
                )
                diagnostics["entropy_trigger_count"] += int(
                    entropy_trigger_counts.sum().item()
                )

                if use_tentative_features:
                    conf_target_counts = torch.clamp(
                        num_transfer_tokens[:, s].to(torch.long) - entropy_trigger_counts,
                        min=0,
                    )
                    tentative_target_counts = (
                        entropy_trigger_counts
                        if enable_tentative_commit
                        else torch.zeros_like(entropy_trigger_counts)
                    )
                    (
                        x,
                        final_mask,
                        tentative_mask,
                        deferred_age,
                        tentative_token_ids,
                        tentative_age,
                        tentative_flip_count,
                        tentative_last_top1,
                        tentative_stable_run,
                        tentative_last_conf,
                        tentative_last_margin,
                    ) = _apply_extended_commit_strategy(
                        tokenizer=self.tokenizer,
                        x=x,
                        x0=x0,
                        confidence=confidence,
                        margin_scores=margin_scores,
                        candidate_mask=candidate_mask,
                        conf_target_counts=conf_target_counts,
                        tentative_target_counts=tentative_target_counts,
                        tentative_candidate_mask=entropy_candidate_mask,
                        step_idx=s,
                        mask_id=mask_id,
                        final_mask=final_mask,
                        tentative_mask=tentative_mask,
                        deferred_age=deferred_age,
                        tentative_token_ids=tentative_token_ids,
                        tentative_age=tentative_age,
                        tentative_flip_count=tentative_flip_count,
                        tentative_last_top1=tentative_last_top1,
                        tentative_stable_run=tentative_stable_run,
                        tentative_last_conf=tentative_last_conf,
                        tentative_last_margin=tentative_last_margin,
                        enable_tentative_commit=enable_tentative_commit,
                        enable_targeted_remask=enable_targeted_remask,
                        tentative_min_hold_steps=tentative_min_hold_steps,
                        tentative_stable_steps=tentative_stable_steps,
                        tentative_max_hold_steps=tentative_max_hold_steps,
                        tentative_final_prob_thresh=tentative_final_prob_thresh,
                        tentative_final_margin_thresh=tentative_final_margin_thresh,
                        remask_rollback_prob_thresh=remask_rollback_prob_thresh,
                        remask_flip_thresh=remask_flip_thresh,
                        diagnostics=diagnostics,
                    )
                else:
                    entropy_transfer = entropy_candidate_mask
                    x[entropy_transfer] = x0[entropy_transfer]
                    confidence_mask = candidate_mask & ~entropy_transfer
                    confidence_counts = torch.clamp(
                        num_transfer_tokens[:, s].to(torch.long) - entropy_trigger_counts,
                        min=0,
                    )
                    transfer_index = select_transfer_positions(
                        confidence=confidence,
                        mask_index=confidence_mask,
                        target_counts=confidence_counts,
                    )
                    x[transfer_index] = x0[transfer_index]
                    if enable_sampler_diagnostics:
                        diagnostics["baseline_finalize_count"] += int(
                            (transfer_index | entropy_transfer).sum().item()
                        )
                        diagnostics["quota_conf_total"] += int(
                            confidence_counts.sum().item()
                        )
                global_step_idx += 1
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        self._last_sampler_diagnostics = diagnostics
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
