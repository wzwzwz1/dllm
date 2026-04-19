import torch
import torch.nn.functional as F

from dllm.core.schedulers import BaseAlphaScheduler

_LOGICAL_CONNECTORS = {
    "because",
    "therefore",
    "thus",
    "since",
    "given",
    "now",
    "to",
    "hence",
    "however",
    "so",
    "then",
    "first",
    "next",
    "finally",
}
_CODE_KEYWORDS = {
    "def",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "return",
    "class",
    "try",
    "except",
    "with",
    "import",
    "from",
    "switch",
    "case",
    "function",
    "var",
    "let",
    "const",
}
_TOKEN_EDGE_PUNCT = ".,;:!?()[]{}\"'"


def _normalize_token(token: str) -> str:
    token = str(token)
    for prefix in ("Ġ", "▁", "Ċ", "ĠĠ"):
        token = token.replace(prefix, "")
    return token.strip().strip(_TOKEN_EDGE_PUNCT).lower()


def _token_has_structure(token: str) -> bool:
    normalized = _normalize_token(token)
    if not normalized:
        return False
    if normalized in _LOGICAL_CONNECTORS or normalized in _CODE_KEYWORDS:
        return True
    return False


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_token_entropy(
    logits: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Compute per-position entropy from logits.

    If `top_k` is set, entropy is approximated on the renormalized top-k slice.
    This keeps the computation cheap enough for inference-time scheduling.
    """
    logits = logits.to(torch.float32)

    if top_k is not None and 0 < top_k < logits.size(-1):
        logits, _ = torch.topk(logits, k=top_k, dim=-1)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_top1_margin(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-position top-1 minus top-2 probability margin.
    """
    probs = F.softmax(logits.to(torch.float32), dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return top2[..., 0]
    return top2[..., 0] - top2[..., 1]


def build_structure_prior_scores(
    tokenizer,
    x: torch.Tensor,
    candidate_mask: torch.Tensor,
    mode: str,
    strength: float,
    *,
    context_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build lightweight structure priors from predicted token ids.

    `x` is expected to be the candidate token ids (e.g. current top-1 prediction).
    When `mode="token_type_with_context"`, `context_tokens` is used to add a
    small bonus for positions next to structural tokens already present in the
    current sequence.
    """
    scores = torch.zeros_like(candidate_mask, dtype=torch.float32)
    if mode == "none" or strength <= 0:
        return scores
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        return scores

    flat_tokens = tokenizer.convert_ids_to_tokens(x.detach().cpu().reshape(-1).tolist())
    structural = torch.zeros_like(candidate_mask, dtype=torch.bool)
    for idx, token in enumerate(flat_tokens):
        structural.view(-1)[idx] = _token_has_structure(token)
    scores = scores + structural.to(torch.float32) * float(strength)

    if mode == "token_type_with_context" and context_tokens is not None:
        context_flat_tokens = tokenizer.convert_ids_to_tokens(
            context_tokens.detach().cpu().reshape(-1).tolist()
        )
        context_structural = torch.zeros_like(candidate_mask, dtype=torch.bool)
        for idx, token in enumerate(context_flat_tokens):
            context_structural.view(-1)[idx] = _token_has_structure(token)

        neighbor_bonus = torch.zeros_like(candidate_mask, dtype=torch.bool)
        neighbor_bonus[:, :-1] |= context_structural[:, 1:]
        neighbor_bonus[:, 1:] |= context_structural[:, :-1]
        scores = scores + neighbor_bonus.to(torch.float32) * float(strength) * 0.5

    return torch.where(candidate_mask, scores, torch.zeros_like(scores))


def build_priority_scores(
    entropy_scores: torch.Tensor,
    confidence: torch.Tensor,
    structure_scores: torch.Tensor,
    age_scores: torch.Tensor,
    *,
    entropy_weight: float = 1.0,
    structure_weight: float = 0.8,
    age_weight: float = 0.4,
    confidence_weight: float = 0.6,
) -> torch.Tensor:
    """
    Compose the structure-aware tentative priority score.
    """
    return (
        entropy_weight * entropy_scores.to(torch.float32)
        + structure_weight * structure_scores.to(torch.float32)
        + age_weight * age_scores.to(torch.float32)
        - confidence_weight * confidence.to(torch.float32)
    )


def compute_entropy_phase_scale(
    step_ratio: float,
    warmup_ratio: float,
    active_end_ratio: float,
    end_ratio: float,
) -> float:
    """
    Compute the phase scale for entropy credit accumulation.

    Warmup:    [0, warmup_ratio)          -> 0.0
    Active:    [warmup_ratio, active_end) -> 1.0
    Cooldown:  [active_end_ratio, end)    -> 0.5
    Off:       [end_ratio, 1.0]           -> 0.0
    """
    if step_ratio < warmup_ratio:
        return 0.0
    if step_ratio < active_end_ratio:
        return 1.0
    if step_ratio < end_ratio:
        return 0.5
    return 0.0


def update_entropy_credit(
    entropy_credit: torch.Tensor,
    phase_scale: float,
    credit_rate: float,
) -> torch.Tensor:
    """
    Accumulate entropy credit for samples currently in the active entropy phase.
    """
    if phase_scale <= 0 or credit_rate <= 0:
        return entropy_credit
    return entropy_credit + float(phase_scale * credit_rate)


def select_entropy_candidate_mask(
    *,
    priority_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    structure_scores: torch.Tensor,
    age_scores: torch.Tensor,
    confidence: torch.Tensor,
    top_candidate_pool: int,
    use_quality_gate: bool,
    confidence_floor: float,
    age_threshold: int,
) -> torch.Tensor:
    """
    Select at most one entropy candidate per row from the top priority pool.
    """
    if priority_scores.shape != candidate_mask.shape:
        raise ValueError("priority_scores and candidate_mask must share the same shape")

    selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if top_candidate_pool <= 0:
        return selected

    neg = torch.finfo(priority_scores.dtype).min
    for row in range(priority_scores.size(0)):
        masked_count = int(candidate_mask[row].sum().item())
        if masked_count == 0:
            continue

        k = min(top_candidate_pool, masked_count)
        row_scores = torch.where(
            candidate_mask[row],
            priority_scores[row].to(torch.float32),
            torch.full_like(priority_scores[row], neg, dtype=torch.float32),
        )
        top_indices = torch.topk(row_scores, k=k).indices

        if use_quality_gate:
            qualified = (
                (structure_scores[row, top_indices] > 0)
                | (age_scores[row, top_indices] >= age_threshold)
                | (confidence[row, top_indices] >= confidence_floor)
            )
        else:
            qualified = torch.ones_like(top_indices, dtype=torch.bool)

        if not qualified.any():
            continue

        chosen_index = top_indices[qualified][0]
        selected[row, chosen_index] = True

    return selected


def compute_entropy_trigger_counts(
    entropy_credit: torch.Tensor,
    candidate_exists: torch.Tensor,
    max_trigger_per_step: int = 1,
) -> torch.Tensor:
    """
    Convert accumulated credit into sparse per-step entropy trigger counts.
    """
    if max_trigger_per_step <= 0:
        return torch.zeros_like(entropy_credit, dtype=torch.long)

    triggerable = (entropy_credit >= 1.0) & candidate_exists.to(torch.bool)
    return triggerable.to(torch.long).clamp(max=max_trigger_per_step)


def update_tentative_stats(
    *,
    current_top1: torch.Tensor,
    current_conf: torch.Tensor,
    current_margin: torch.Tensor,
    tentative_mask: torch.Tensor,
    tentative_token_ids: torch.Tensor,
    tentative_age: torch.Tensor,
    tentative_flip_count: torch.Tensor,
    tentative_last_top1: torch.Tensor,
    tentative_stable_run: torch.Tensor,
    tentative_last_conf: torch.Tensor,
    tentative_last_margin: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Update per-position statistics for tokens currently in TENTATIVE state.
    """
    active = tentative_mask
    token_changed = active & (current_top1 != tentative_last_top1)

    tentative_age = torch.where(active, tentative_age + 1, tentative_age)
    tentative_flip_count = torch.where(
        active, tentative_flip_count + token_changed.to(torch.long), tentative_flip_count
    )
    tentative_stable_run = torch.where(
        active,
        torch.where(
            token_changed,
            torch.ones_like(tentative_stable_run),
            tentative_stable_run + 1,
        ),
        tentative_stable_run,
    )
    tentative_token_ids = torch.where(active, current_top1, tentative_token_ids)
    tentative_last_top1 = torch.where(active, current_top1, tentative_last_top1)
    tentative_last_conf = torch.where(active, current_conf, tentative_last_conf)
    tentative_last_margin = torch.where(active, current_margin, tentative_last_margin)

    return (
        tentative_token_ids,
        tentative_age,
        tentative_flip_count,
        tentative_last_top1,
        tentative_stable_run,
        tentative_last_conf,
        tentative_last_margin,
    )


def compute_tentative_finalize_mask(
    *,
    tentative_mask: torch.Tensor,
    tentative_age: torch.Tensor,
    tentative_stable_run: torch.Tensor,
    tentative_last_conf: torch.Tensor,
    tentative_last_margin: torch.Tensor,
    min_hold_steps: int,
    stable_steps: int,
    final_prob_thresh: float,
    final_margin_thresh: float,
) -> torch.Tensor:
    """
    Compute which TENTATIVE positions are stable enough to finalize.
    """
    stable_enough = (tentative_age >= min_hold_steps) & (
        tentative_stable_run >= stable_steps
    )
    high_prob = tentative_last_conf >= final_prob_thresh
    high_margin = tentative_last_margin >= final_margin_thresh
    return tentative_mask & (stable_enough | high_prob | high_margin)


def compute_tentative_rollback_mask(
    *,
    tentative_mask: torch.Tensor,
    tentative_age: torch.Tensor,
    tentative_flip_count: torch.Tensor,
    tentative_last_conf: torch.Tensor,
    tentative_stable_run: torch.Tensor,
    max_hold_steps: int,
    rollback_prob_thresh: float,
    flip_thresh: int,
    stable_steps: int,
) -> torch.Tensor:
    """
    Compute which TENTATIVE positions should be remasked.
    """
    aged_unstable = (tentative_age >= max_hold_steps) & (
        tentative_stable_run < stable_steps
    )
    too_many_flips = tentative_flip_count >= flip_thresh
    low_confidence = tentative_last_conf <= rollback_prob_thresh
    return tentative_mask & (aged_unstable | too_many_flips | low_confidence)


def select_transfer_positions(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    target_counts: torch.Tensor,
    *,
    entropy_scores: torch.Tensor | None = None,
    entropy_first_k: int = 0,
) -> torch.Tensor:
    """
    Select masked positions to update this step.

    By default, positions are chosen by descending confidence. When
    `entropy_scores` and `entropy_first_k` are provided, each row reserves up to
    `entropy_first_k` slots for the highest-entropy masked positions and fills
    the remaining budget with the highest-confidence masked positions.
    """
    if confidence.shape != mask_index.shape:
        raise ValueError(
            f"confidence shape {confidence.shape} must match mask_index {mask_index.shape}"
        )

    if target_counts.dim() != 1 or target_counts.size(0) != confidence.size(0):
        raise ValueError(
            f"target_counts shape {target_counts.shape} must be [B], got batch={confidence.size(0)}"
        )

    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
    neg = torch.finfo(confidence.dtype).min
    target_counts = target_counts.to(device=confidence.device, dtype=torch.long)

    for row in range(confidence.size(0)):
        masked_count = int(mask_index[row].sum().item())
        if masked_count == 0:
            continue

        total_budget = min(int(target_counts[row].item()), masked_count)
        if total_budget <= 0:
            continue

        reserved_entropy = 0
        if entropy_scores is not None and entropy_first_k > 0:
            reserved_entropy = min(entropy_first_k, total_budget)
            entropy_row = torch.where(
                mask_index[row],
                entropy_scores[row].to(confidence.dtype),
                torch.full_like(confidence[row], neg),
            )
            _, entropy_idx = torch.topk(entropy_row, k=reserved_entropy)
            transfer_index[row, entropy_idx] = True

        remaining_budget = total_budget - reserved_entropy
        if remaining_budget <= 0:
            continue

        confidence_row = confidence[row].clone()
        confidence_row[~mask_index[row]] = neg
        confidence_row[transfer_index[row]] = neg
        _, confidence_idx = torch.topk(confidence_row, k=remaining_budget)
        transfer_index[row, confidence_idx] = True

    return transfer_index
