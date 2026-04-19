"""
Unit tests for dllm.core.eval.base helper utilities.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /Users/wz/code/dllm/scripts/tests/test_eval_base.py -v
"""

import pytest

from dllm.core.eval.base import BaseEvalHarness, BaseEvalConfig


class TestBuildConfig:
    def test_build_config_from_source(self):
        source = BaseEvalConfig(pretrained="my/model", device="cuda", batch_size=4)
        out = BaseEvalHarness._build_config(BaseEvalConfig, source, {})
        assert out.pretrained == "my/model"
        assert out.device == "cuda"
        assert out.batch_size == 4

    def test_kwargs_override_source(self):
        source = BaseEvalConfig(pretrained="a", device="cuda", batch_size=1)
        out = BaseEvalHarness._build_config(
            BaseEvalConfig, source, {"batch_size": 8, "device": "cpu"}
        )
        assert out.pretrained == "a"
        assert out.device == "cpu"
        assert out.batch_size == 8

    def test_partial_kwargs_fill_remaining_from_source(self):
        source = BaseEvalConfig(pretrained="base", device="cuda", batch_size=2)
        out = BaseEvalHarness._build_config(BaseEvalConfig, source, {"batch_size": 16})
        assert out.pretrained == "base"
        assert out.device == "cuda"
        assert out.batch_size == 16


class TestEvalLoggingHelpers:
    def test_extract_question_text_prefers_last_question_block(self):
        context = (
            "Q: 1+1?\nA: 2\n\n"
            "Q: If Tom has 3 apples and buys 2 more, how many apples does he have?\nA:"
        )
        out = BaseEvalHarness._extract_question_text(context)
        assert out.startswith("Q:")
        assert "Tom has 3 apples" in out
        assert "1+1" not in out

    def test_extract_final_answer_prefers_hash_marker(self):
        text = "Let's think step by step.\n#### 42"
        assert BaseEvalHarness._extract_final_answer(text) == "42"

    def test_extract_final_answer_falls_back_to_last_nonempty_line(self):
        text = "Reasoning...\nFinal line"
        assert BaseEvalHarness._extract_final_answer(text) == "Final line"

    def test_compress_trace_keeps_first_and_last(self):
        trace = [{"step": i, "text": str(i)} for i in range(10)]
        out = BaseEvalHarness._compress_trace(trace, max_steps=4)
        assert len(out) == 4
        assert out[0]["step"] == 0
        assert out[-1]["step"] == 9
