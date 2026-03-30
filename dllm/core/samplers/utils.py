import torch
import torch.nn.functional as F

from dllm.core.schedulers import BaseAlphaScheduler


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
