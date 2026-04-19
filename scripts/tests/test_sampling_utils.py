"""
Unit tests for dllm sampling utilities: sample_trim, infill_trim, add_gumbel_noise, get_num_transfer_tokens.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_sampling_utils.py -v
"""

import pytest
import torch

from types import SimpleNamespace

from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.utils.sampling import sample_trim, infill_trim
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
from dllm.core.schedulers import LinearAlphaScheduler


def _make_mock_tokenizer(
    pad_token_id=0,
    eos_token_id=1,
    eot_token_id=None,
    eos_token="</s>",
    eot_token=None,
    mask_token_id=2,
):
    """Minimal tokenizer-like object for testing trim functions."""
    tok = SimpleNamespace()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.eot_token_id = eot_token_id
    tok.eos_token = eos_token
    tok.eot_token = eot_token
    tok.mask_token_id = mask_token_id

    def decode(ids, skip_special_tokens=True):
        # Treat 0 as pad, 1 as eos; strip them if skip_special_tokens
        ids = list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in (0, 1)]
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    tok.decode = decode
    tok.id_to_token = {
        0: "<pad>",
        1: "</s>",
        2: "<mask>",
        3: "=",
        4: "because",
        5: "alpha",
        6: "beta",
        7: "word",
        8: "(",
        9: ")",
        10: "prompt",
        11: "Therefore",
        12: "return",
    }

    def convert_ids_to_tokens(ids):
        return [tok.id_to_token.get(i, str(i)) for i in ids]

    tok.convert_ids_to_tokens = convert_ids_to_tokens
    return tok


class _MockModel:
    def __init__(self, logits_by_call):
        self.logits_by_call = logits_by_call
        self.device = torch.device("cpu")
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None):
        idx = min(self.call_count, len(self.logits_by_call) - 1)
        logits = self.logits_by_call[idx].clone().to(input_ids.device)
        self.call_count += 1
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# sample_trim
# ---------------------------------------------------------------------------


class TestSampleTrim:
    def test_no_eos_returns_full_generation(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [10, 11, 12, 13, 14]  # prompt len 2 -> gen 3
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # Decoded gen part: indices 2:5 -> [12,13,14]
        assert out[0] == "mno"  # 12->m, 13->n, 14->o

    def test_stops_at_first_eos_after_prompt(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [10, 11, 12, 1, 99]  # eos at index 3
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # gen = 12 only (then eos); decode skips 1
        assert out[0] == "m"  # gen [12], eos at 3

    def test_left_padding_skipped(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [0, 0, 10, 11, 12]  # 2 pad, prompt 10,11 -> start at 2, gen 12
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # After skipping pads: [10,11,12], start=2, end=3 -> gen [12]
        assert out[0] == "m"

    def test_batch_multiple_sequences(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_list = [[1, 2, 3, 4], [5, 6, 7, 8]]
        input_list = [[1, 2], [5, 6]]
        out = sample_trim(tok, seq_list, input_list)
        assert len(out) == 2
        assert out[0] == "de"  # gen [3,4] -> d,e
        assert out[1] == "hi"  # gen [7,8] -> h,i


# ---------------------------------------------------------------------------
# infill_trim
# ---------------------------------------------------------------------------


class TestInfillTrim:
    def test_extracts_masked_positions_only(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1, mask_token_id=99)
        # input has mask at positions that get filled; full seq has values there
        prompt = [10, 99, 11, 99]  # two masks
        full = [10, 20, 11, 21]    # infill 20, 21
        out = infill_trim(tok, [full], [prompt])
        assert len(out) == 1
        # infill tokens = [20, 21] -> decode
        assert out[0] == "uv"  # 20->u, 21->v

    def test_infill_stops_at_eos(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1, mask_token_id=99)
        prompt = [99, 99, 99]
        full = [2, 3, 1]  # eos at index 2 in infill; gen before = [2,3]
        out = infill_trim(tok, [full], [prompt])
        assert len(out) == 1
        assert out[0] == "cd"  # 2->c, 3->d (chr(97+i))


# ---------------------------------------------------------------------------
# add_gumbel_noise
# ---------------------------------------------------------------------------


class TestAddGumbelNoise:
    def test_temperature_zero_returns_unchanged(self):
        logits = torch.randn(2, 10)
        out = add_gumbel_noise(logits, temperature=0.0)
        assert out is logits
        assert torch.allclose(out.float(), logits.float())

    def test_temperature_positive_changes_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = add_gumbel_noise(logits, temperature=0.5)
        assert out.shape == logits.shape
        assert out.dtype == torch.float64
        # Random; at least check no nan/inf
        assert torch.isfinite(out).all()

    def test_output_dtype_float64(self):
        logits = torch.randn(3, 5, dtype=torch.float32)
        out = add_gumbel_noise(logits, temperature=0.3)
        assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# get_num_transfer_tokens
# ---------------------------------------------------------------------------


class TestGetNumTransferTokens:
    def test_deterministic_single_sample(self):
        scheduler = LinearAlphaScheduler()
        # 4 masked tokens, 3 steps
        mask_index = torch.tensor([[True, True, True, True]])
        out = get_num_transfer_tokens(mask_index, steps=3, scheduler=scheduler, stochastic=False)
        assert out.shape[0] == 1
        assert out.dtype == torch.int64
        # Sum of transfers should not exceed 4
        assert out.sum().item() <= 4

    def test_batch(self):
        scheduler = LinearAlphaScheduler()
        mask_index = torch.tensor([
            [True, True, False, False],
            [True, True, True, True],
        ])
        out = get_num_transfer_tokens(mask_index, steps=4, scheduler=scheduler, stochastic=False)
        assert out.shape[0] == 2
        assert out.dtype == torch.int64

    def test_stochastic_same_shape(self):
        scheduler = LinearAlphaScheduler()
        mask_index = torch.tensor([[True] * 10])
        out = get_num_transfer_tokens(mask_index, steps=5, scheduler=scheduler, stochastic=True)
        assert out.shape[0] == 1
        assert out.dtype == torch.int64
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# entropy-aware scheduling utilities
# ---------------------------------------------------------------------------


class TestGetTokenEntropy:
    def test_matches_manual_binary_entropy(self):
        logits = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]], dtype=torch.float32)
        out = get_token_entropy(logits)
        assert out.shape == torch.Size([1, 2])
        # First row is uniform => ln(2)
        assert out[0, 0].item() == pytest.approx(0.6931, rel=1e-3)
        # Second row is much lower entropy than the first
        assert out[0, 1].item() < out[0, 0].item()

    def test_topk_entropy_runs(self):
        logits = torch.randn(2, 3, 11)
        out = get_token_entropy(logits, top_k=4)
        assert out.shape == torch.Size([2, 3])
        assert torch.isfinite(out).all()


class TestPriorityHelpers:
    def test_get_top1_margin(self):
        logits = torch.tensor([[[3.0, 1.0, 0.0]]], dtype=torch.float32)
        out = get_top1_margin(logits)
        assert out.shape == torch.Size([1, 1])
        assert out.item() > 0.5

    def test_structure_prior_none_returns_zero(self):
        tokenizer = _make_mock_tokenizer()
        x = torch.tensor([[3, 7, 4]], dtype=torch.long)
        mask = torch.tensor([[True, True, False]])
        out = build_structure_prior_scores(tokenizer, x, mask, "none", 1.0)
        assert out.tolist() == [[0.0, 0.0, 0.0]]

    def test_structure_prior_detects_structural_tokens(self):
        tokenizer = _make_mock_tokenizer()
        x = torch.tensor([[11, 7, 4]], dtype=torch.long)
        mask = torch.tensor([[True, True, True]])
        out = build_structure_prior_scores(tokenizer, x, mask, "token_type", 1.0)
        assert out[0, 0].item() > out[0, 1].item()
        assert out[0, 2].item() > out[0, 1].item()

    def test_structure_prior_ignores_symbols(self):
        tokenizer = _make_mock_tokenizer()
        x = torch.tensor([[3, 7, 12]], dtype=torch.long)
        mask = torch.tensor([[True, True, True]])
        out = build_structure_prior_scores(tokenizer, x, mask, "token_type", 1.0)
        assert out[0, 0].item() == 0.0
        assert out[0, 2].item() > out[0, 1].item()

    def test_build_priority_scores_can_disable_components(self):
        entropy = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        confidence = torch.tensor([[0.3, 0.3]], dtype=torch.float32)
        structure = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        age = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
        out = build_priority_scores(
            entropy,
            confidence,
            structure,
            age,
            entropy_weight=1.0,
            structure_weight=0.0,
            age_weight=0.0,
            confidence_weight=0.0,
        )
        assert out.tolist() == [[0.8, 0.2]]

    def test_update_tentative_stats_tracks_flip_and_stability(self):
        tentative_mask = torch.tensor([[True, False]])
        current_top1 = torch.tensor([[5, 6]], dtype=torch.long)
        current_conf = torch.tensor([[0.7, 0.1]], dtype=torch.float32)
        current_margin = torch.tensor([[0.4, 0.1]], dtype=torch.float32)
        stats = update_tentative_stats(
            current_top1=current_top1,
            current_conf=current_conf,
            current_margin=current_margin,
            tentative_mask=tentative_mask,
            tentative_token_ids=torch.zeros((1, 2), dtype=torch.long),
            tentative_age=torch.zeros((1, 2), dtype=torch.long),
            tentative_flip_count=torch.zeros((1, 2), dtype=torch.long),
            tentative_last_top1=torch.tensor([[4, -1]], dtype=torch.long),
            tentative_stable_run=torch.zeros((1, 2), dtype=torch.long),
            tentative_last_conf=torch.zeros((1, 2), dtype=torch.float32),
            tentative_last_margin=torch.zeros((1, 2), dtype=torch.float32),
        )
        assert stats[1][0, 0].item() == 1
        assert stats[2][0, 0].item() == 1
        assert stats[4][0, 0].item() == 1

    def test_finalize_mask_supports_stable_or_high_confidence(self):
        finalize_mask = compute_tentative_finalize_mask(
            tentative_mask=torch.tensor([[True, True, True]]),
            tentative_age=torch.tensor([[1, 0, 0]], dtype=torch.long),
            tentative_stable_run=torch.tensor([[2, 1, 1]], dtype=torch.long),
            tentative_last_conf=torch.tensor([[0.6, 0.9, 0.3]], dtype=torch.float32),
            tentative_last_margin=torch.tensor([[0.2, 0.1, 0.4]], dtype=torch.float32),
            min_hold_steps=1,
            stable_steps=2,
            final_prob_thresh=0.82,
            final_margin_thresh=0.35,
        )
        assert finalize_mask.tolist() == [[True, True, True]]

    def test_rollback_mask_supports_age_flip_and_low_confidence(self):
        rollback_mask = compute_tentative_rollback_mask(
            tentative_mask=torch.tensor([[True, True, True]]),
            tentative_age=torch.tensor([[3, 1, 1]], dtype=torch.long),
            tentative_flip_count=torch.tensor([[0, 2, 0]], dtype=torch.long),
            tentative_last_conf=torch.tensor([[0.6, 0.6, 0.3]], dtype=torch.float32),
            tentative_stable_run=torch.tensor([[1, 1, 1]], dtype=torch.long),
            max_hold_steps=3,
            rollback_prob_thresh=0.45,
            flip_thresh=2,
            stable_steps=2,
        )
        assert rollback_mask.tolist() == [[True, True, True]]

    def test_entropy_phase_scale_segments(self):
        assert compute_entropy_phase_scale(0.01, 0.05, 0.20, 0.30) == 0.0
        assert compute_entropy_phase_scale(0.10, 0.05, 0.20, 0.30) == 1.0
        assert compute_entropy_phase_scale(0.25, 0.05, 0.20, 0.30) == 0.5
        assert compute_entropy_phase_scale(0.40, 0.05, 0.20, 0.30) == 0.0

    def test_update_entropy_credit_accumulates_by_phase_scale(self):
        credit = torch.tensor([0.0, 0.5], dtype=torch.float32)
        out = update_entropy_credit(credit, phase_scale=0.5, credit_rate=0.4)
        assert out.tolist() == pytest.approx([0.2, 0.7], rel=1e-5)

    def test_select_entropy_candidate_mask_applies_quality_gate(self):
        priority = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        candidate_mask = torch.tensor([[True, True, True]])
        structure = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        age = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
        confidence = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32)
        out = select_entropy_candidate_mask(
            priority_scores=priority,
            candidate_mask=candidate_mask,
            structure_scores=structure,
            age_scores=age,
            confidence=confidence,
            top_candidate_pool=2,
            use_quality_gate=True,
            confidence_floor=0.15,
            age_threshold=2,
        )
        assert out.tolist() == [[False, True, False]]

    def test_select_entropy_candidate_mask_without_quality_gate_uses_top_priority(self):
        priority = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)
        candidate_mask = torch.tensor([[True, True, True]])
        zeros = torch.zeros_like(priority)
        out = select_entropy_candidate_mask(
            priority_scores=priority,
            candidate_mask=candidate_mask,
            structure_scores=zeros,
            age_scores=zeros,
            confidence=zeros,
            top_candidate_pool=3,
            use_quality_gate=False,
            confidence_floor=0.15,
            age_threshold=2,
        )
        assert out.tolist() == [[True, False, False]]

    def test_entropy_trigger_counts_require_credit_and_candidate(self):
        credit = torch.tensor([0.6, 1.1, 1.4], dtype=torch.float32)
        candidate_exists = torch.tensor([True, False, True])
        out = compute_entropy_trigger_counts(
            credit, candidate_exists, max_trigger_per_step=1
        )
        assert out.tolist() == [0, 0, 1]


class TestSelectTransferPositions:
    def test_confidence_only_matches_topk_budget(self):
        confidence = torch.tensor([[0.2, 0.9, 0.7, 0.1]], dtype=torch.float32)
        mask_index = torch.tensor([[True, True, True, False]])
        counts = torch.tensor([2], dtype=torch.long)

        out = select_transfer_positions(confidence, mask_index, counts)
        assert out.tolist() == [[False, True, True, False]]

    def test_entropy_budget_reserves_high_entropy_slot(self):
        confidence = torch.tensor([[0.95, 0.90, 0.10, 0.05]], dtype=torch.float32)
        entropy = torch.tensor([[0.05, 0.10, 0.99, 0.20]], dtype=torch.float32)
        mask_index = torch.tensor([[True, True, True, False]])
        counts = torch.tensor([2], dtype=torch.long)

        out = select_transfer_positions(
            confidence,
            mask_index,
            counts,
            entropy_scores=entropy,
            entropy_first_k=1,
        )

        # One slot goes to the highest-entropy masked position (idx=2),
        # the other remains the highest-confidence masked position (idx=0).
        assert out.tolist() == [[True, False, True, False]]

    def test_entropy_budget_never_exceeds_available_masks(self):
        confidence = torch.tensor([[0.8, 0.1, 0.2]], dtype=torch.float32)
        entropy = torch.tensor([[0.3, 0.9, 0.1]], dtype=torch.float32)
        mask_index = torch.tensor([[True, False, True]])
        counts = torch.tensor([5], dtype=torch.long)

        out = select_transfer_positions(
            confidence,
            mask_index,
            counts,
            entropy_scores=entropy,
            entropy_first_k=3,
        )

        assert out.sum().item() == 2
        assert out.tolist() == [[True, False, True]]


class TestMDLMSamplerExtensions:
    def _make_sampler(self, logits_by_call):
        tokenizer = _make_mock_tokenizer(mask_token_id=2)
        model = _MockModel(logits_by_call=logits_by_call)
        return MDLMSampler(model=model, tokenizer=tokenizer)

    def test_default_flags_preserve_baseline_output(self):
        logits = torch.full((1, 3, 12), -10.0, dtype=torch.float32)
        logits[0, 1, 5] = 5.0
        logits[0, 2, 6] = 5.0
        inputs = [[10]]

        sampler_a = self._make_sampler([logits, logits])
        sampler_b = self._make_sampler([logits, logits])

        out_a = sampler_a.sample(
            inputs,
            MDLMSamplerConfig(steps=4, max_new_tokens=2, block_size=2),
            return_dict=False,
        )
        out_b = sampler_b.sample(
            inputs,
            MDLMSamplerConfig(
                steps=4,
                max_new_tokens=2,
                block_size=2,
                enable_entropy_priority=False,
                enable_tentative_commit=False,
                enable_targeted_remask=False,
                enable_structure_priority=False,
                enable_priority_age_bonus=False,
            ),
            return_dict=False,
        )
        assert torch.equal(out_a, out_b)

    def test_tentative_commit_enters_without_rollback_when_disabled(self):
        logits_1 = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits_2 = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits_1[0, 1, 5] = 1.0
        logits_1[0, 2, 6] = 1.0
        logits_2[0, 1, 5] = 3.0
        logits_2[0, 2, 6] = 3.0

        sampler = self._make_sampler([logits_1, logits_2, logits_2])
        sampler.sample(
            [[10]],
            MDLMSamplerConfig(
                steps=4,
                max_new_tokens=2,
                block_size=2,
                enable_entropy_priority=True,
                enable_entropy_credit_scheduler=True,
                entropy_credit_rate=1.0,
                entropy_warmup_ratio=0.0,
                entropy_active_end_ratio=1.0,
                entropy_end_ratio=1.0,
                enable_tentative_commit=True,
                enable_targeted_remask=False,
                entropy_use_quality_gate=False,
                enable_sampler_diagnostics=True,
            ),
            return_dict=False,
        )
        diagnostics = sampler._last_sampler_diagnostics
        assert diagnostics["tentative_enter_count"] > 0
        assert diagnostics["tentative_rollback_count"] == 0
        assert diagnostics["entropy_trigger_count"] > 0

    def test_targeted_remask_can_rollback_tentative_tokens(self):
        logits_1 = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits_2 = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits_3 = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits_1[0, 1, 5] = 1.0
        logits_1[0, 2, 6] = 1.0
        logits_2[0, 1, 7] = 0.2
        logits_2[0, 2, 6] = 3.0
        logits_3[0, 1, 5] = 3.0
        logits_3[0, 2, 6] = 3.0

        sampler = self._make_sampler([logits_1, logits_2, logits_3, logits_3])
        sampler.sample(
            [[10]],
            MDLMSamplerConfig(
                steps=4,
                max_new_tokens=2,
                block_size=2,
                enable_entropy_priority=True,
                enable_entropy_credit_scheduler=True,
                entropy_credit_rate=1.0,
                entropy_warmup_ratio=0.0,
                entropy_active_end_ratio=1.0,
                entropy_end_ratio=1.0,
                enable_tentative_commit=True,
                enable_targeted_remask=True,
                entropy_use_quality_gate=False,
                remask_flip_thresh=1,
                remask_rollback_prob_thresh=0.95,
                enable_sampler_diagnostics=True,
            ),
            return_dict=False,
        )
        diagnostics = sampler._last_sampler_diagnostics
        assert diagnostics["tentative_enter_count"] > 0
        assert diagnostics["tentative_rollback_count"] > 0

    def test_entropy_only_uses_credit_trigger_instead_of_fixed_every_step(self):
        logits = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits[0, 1, 11] = 2.0
        logits[0, 2, 5] = 2.0
        sampler = self._make_sampler([logits, logits, logits, logits])
        sampler.sample(
            [[10]],
            MDLMSamplerConfig(
                steps=4,
                max_new_tokens=2,
                block_size=2,
                enable_entropy_priority=True,
                enable_entropy_credit_scheduler=True,
                entropy_credit_rate=0.4,
                entropy_warmup_ratio=0.0,
                entropy_active_end_ratio=1.0,
                entropy_end_ratio=1.0,
                enable_tentative_commit=False,
                entropy_use_quality_gate=False,
                enable_sampler_diagnostics=True,
            ),
            return_dict=False,
        )
        diagnostics = sampler._last_sampler_diagnostics
        # 4 steps * 0.4 credit => sparse trigger once instead of every step.
        assert diagnostics["entropy_trigger_count"] == 1
        assert diagnostics["entropy_finalize_count"] == 1
        assert diagnostics["baseline_finalize_count"] == 1
        assert diagnostics["finalized_token_count"] == 2

    def test_structure_priority_can_enable_credit_candidate_selection(self):
        logits = torch.full((1, 3, 12), -8.0, dtype=torch.float32)
        logits[0, 1, 5] = 2.5  # non-structural token with higher raw score
        logits[0, 2, 11] = 2.0  # structural token "Therefore"
        sampler = self._make_sampler([logits, logits])
        sampler.sample(
            [[10]],
            MDLMSamplerConfig(
                steps=2,
                max_new_tokens=2,
                block_size=2,
                enable_entropy_priority=True,
                enable_entropy_credit_scheduler=True,
                entropy_credit_rate=1.0,
                entropy_warmup_ratio=0.0,
                entropy_active_end_ratio=1.0,
                entropy_end_ratio=1.0,
                enable_tentative_commit=True,
                enable_structure_priority=True,
                structure_prior_mode="token_type",
                entropy_use_quality_gate=True,
                entropy_conf_floor=0.99,
                enable_sampler_diagnostics=True,
            ),
            return_dict=False,
        )
        diagnostics = sampler._last_sampler_diagnostics
        assert diagnostics["tentative_enter_count"] > 0
        assert diagnostics["finalized_token_count"] >= diagnostics["tentative_finalize_count"]

    def test_infill_credit_scheduler_runs_tentative_path(self):
        logits = torch.full((1, 3, 12), -6.0, dtype=torch.float32)
        logits[0, 1, 11] = 2.0
        logits[0, 2, 5] = 2.0
        sampler = self._make_sampler([logits, logits, logits])
        sampler.infill(
            [[10, 2, 2]],
            MDLMSamplerConfig(
                steps=3,
                block_size=3,
                enable_entropy_priority=True,
                enable_entropy_credit_scheduler=True,
                entropy_credit_rate=1.0,
                entropy_warmup_ratio=0.0,
                entropy_active_end_ratio=1.0,
                entropy_end_ratio=1.0,
                enable_tentative_commit=True,
                entropy_use_quality_gate=False,
                enable_sampler_diagnostics=True,
            ),
            return_dict=False,
        )
        diagnostics = sampler._last_sampler_diagnostics
        assert diagnostics["tentative_enter_count"] > 0
