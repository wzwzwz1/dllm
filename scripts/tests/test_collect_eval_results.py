"""
Tests for eval log parsing helpers used by sweep leaderboard generation.

Run from repo root:
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm
  pytest /Users/wz/code/dllm/scripts/tests/test_collect_eval_results.py -v
"""

from pathlib import Path

from scripts.collect_eval_results import parse_eval_log


def test_parse_eval_log_extracts_flexible_and_strict_metrics(tmp_path: Path):
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "\n".join(
            [
                "llada (pretrained=model,max_new_tokens=256,steps=128,block_size=32,enable_entropy_priority=True), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1",
                "Generating...: 100%|██████████| 100/100 [12:34<00:00,  7.89it/s]",
                "|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|",
                "|---|---:|---|---:|---|---|---:|---|---:|",
                "|gsm8k_cot|1|flexible-extract|5|exact_match|↑|0.4200|±|0.0490|",
                "| | |strict-match|5|exact_match|↑|0.3900|±|0.0480|",
            ]
        ),
        encoding="utf-8",
    )

    result = parse_eval_log(log_path)

    assert result.limit == 100
    assert result.steps == "128"
    assert result.block_size == "32"
    assert result.flexible_exact_match == 0.42
    assert result.flexible_stderr == 0.049
    assert result.strict_exact_match == 0.39
    assert result.strict_stderr == 0.048
    assert result.duration == "12:34"


def test_parse_eval_log_tolerates_ansi_sequences(tmp_path: Path):
    log_path = tmp_path / "eval_ansi.log"
    log_path.write_text(
        "\n".join(
            [
                "\u001b[32mllada (pretrained=model,steps=128,block_size=32), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1\u001b[0m",
                "\u001b[33m|gsm8k_cot|1|flexible-extract|5|exact_match|↑|0.5000|±|0.0500|\u001b[0m",
                "\u001b[33m| | |strict-match|5|exact_match|↑|0.4700|±|0.0490|\u001b[0m",
            ]
        ),
        encoding="utf-8",
    )

    result = parse_eval_log(log_path)

    assert result.flexible_exact_match == 0.5
    assert result.strict_exact_match == 0.47
