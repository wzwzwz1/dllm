"""
Update manifest and leaderboard files for entropy-priority grid sweeps.

Run from the server:
  source ~/.zshrc
  source /home/wangzhe/miniconda3/etc/profile.d/conda.sh
  conda activate dllm
  cd /disk/wangzhe/dllm
  export PYTHONPATH=.:$PYTHONPATH
  python /disk/wangzhe/dllm/scripts/update_entropy_grid_reports.py \
    --sweep-root /disk/wangzhe/dllm/.logs/sweeps/20260419-210000-gsm8k-cot-limit200-entropy-grid72
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_eval_results import parse_eval_log


@dataclass
class RunRecord:
    run_id: str
    slug: str
    status: str
    exit_code: int | None
    assigned_gpu: int | None
    credit_rate: float | None
    warmup_ratio: float | None
    active_end_ratio: float | None
    end_ratio: float | None
    entropy_top_k: int | None
    flexible_exact_match: float | None
    strict_exact_match: float | None
    flexible_stderr: float | None
    strict_stderr: float | None
    duration: str | None
    duration_seconds: int | None
    log_path: str
    generation_records_path: str
    meta_path: str
    note: str


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration_to_seconds(value: str | None) -> int | None:
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except ValueError:
        return None
    return None


def _load_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_meta_metrics(meta_path: Path) -> dict[str, Any]:
    payload = _load_meta(meta_path)
    log_path = Path(str(payload.get("eval_log_path", "")))

    metrics: dict[str, Any] = {
        "flexible_exact_match": None,
        "strict_exact_match": None,
        "flexible_stderr": None,
        "strict_stderr": None,
        "duration": None,
    }
    note = str(payload.get("note", "")).strip()

    if log_path.exists():
        parsed = parse_eval_log(log_path)
        metrics = {
            "flexible_exact_match": parsed.flexible_exact_match,
            "strict_exact_match": parsed.strict_exact_match,
            "flexible_stderr": parsed.flexible_stderr,
            "strict_stderr": parsed.strict_stderr,
            "duration": parsed.duration,
        }
        if payload.get("status") == "completed" and parsed.flexible_exact_match is None:
            note = "completed run without parsed flexible exact match"
    elif payload.get("status") == "completed":
        note = "completed marker exists but eval.log is missing"

    payload["metrics"] = metrics
    payload["note"] = note
    _write_meta(meta_path, payload)
    return payload


def _meta_to_record(meta: dict[str, Any], meta_path: Path) -> RunRecord:
    params = meta.get("params", {})
    metrics = meta.get("metrics", {})
    duration = metrics.get("duration")
    return RunRecord(
        run_id=str(meta.get("run_id", meta_path.parent.name)),
        slug=str(meta.get("slug", meta_path.parent.name)),
        status=str(meta.get("status", "unknown")),
        exit_code=_safe_int(meta.get("exit_code")),
        assigned_gpu=_safe_int(meta.get("assigned_gpu")),
        credit_rate=_safe_float(params.get("entropy_credit_rate")),
        warmup_ratio=_safe_float(params.get("entropy_warmup_ratio")),
        active_end_ratio=_safe_float(params.get("entropy_active_end_ratio")),
        end_ratio=_safe_float(params.get("entropy_end_ratio")),
        entropy_top_k=_safe_int(params.get("entropy_top_k")),
        flexible_exact_match=_safe_float(metrics.get("flexible_exact_match")),
        strict_exact_match=_safe_float(metrics.get("strict_exact_match")),
        flexible_stderr=_safe_float(metrics.get("flexible_stderr")),
        strict_stderr=_safe_float(metrics.get("strict_stderr")),
        duration=duration,
        duration_seconds=_duration_to_seconds(duration),
        log_path=str(meta.get("eval_log_path", "")),
        generation_records_path=str(meta.get("generation_records_path", "")),
        meta_path=str(meta_path),
        note=str(meta.get("note", "")),
    )


def _scan_records(sweep_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for meta_path in sorted((sweep_root / "runs").glob("*/meta.json")):
        payload = _refresh_meta_metrics(meta_path)
        records.append(_meta_to_record(payload, meta_path))
    return records


def _write_grid_manifest(records: list[RunRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "slug",
                "status",
                "exit_code",
                "assigned_gpu",
                "entropy_credit_rate",
                "entropy_warmup_ratio",
                "entropy_active_end_ratio",
                "entropy_end_ratio",
                "entropy_top_k",
                "flexible_exact_match",
                "strict_exact_match",
                "flexible_stderr",
                "strict_stderr",
                "duration",
                "eval_log_path",
                "generation_records_path",
                "meta_path",
                "note",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.run_id,
                    record.slug,
                    record.status,
                    record.exit_code,
                    record.assigned_gpu,
                    record.credit_rate,
                    record.warmup_ratio,
                    record.active_end_ratio,
                    record.end_ratio,
                    record.entropy_top_k,
                    record.flexible_exact_match,
                    record.strict_exact_match,
                    record.flexible_stderr,
                    record.strict_stderr,
                    record.duration,
                    record.log_path,
                    record.generation_records_path,
                    record.meta_path,
                    record.note,
                ]
            )


def _sort_leaderboard(records: list[RunRecord]) -> list[RunRecord]:
    completed = [
        record
        for record in records
        if record.status == "completed" and record.flexible_exact_match is not None
    ]
    return sorted(
        completed,
        key=lambda item: (
            -(item.flexible_exact_match or -1.0),
            -(item.strict_exact_match or -1.0),
            item.duration_seconds
            if item.duration_seconds is not None
            else 10**12,
            item.run_id,
        ),
    )


def _write_leaderboard_csv(records: list[RunRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "run_id",
                "slug",
                "assigned_gpu",
                "entropy_credit_rate",
                "entropy_warmup_ratio",
                "entropy_active_end_ratio",
                "entropy_end_ratio",
                "entropy_top_k",
                "flexible_exact_match",
                "strict_exact_match",
                "flexible_stderr",
                "strict_stderr",
                "duration",
                "eval_log_path",
                "generation_records_path",
            ]
        )
        for rank, record in enumerate(records, start=1):
            writer.writerow(
                [
                    rank,
                    record.run_id,
                    record.slug,
                    record.assigned_gpu,
                    record.credit_rate,
                    record.warmup_ratio,
                    record.active_end_ratio,
                    record.end_ratio,
                    record.entropy_top_k,
                    record.flexible_exact_match,
                    record.strict_exact_match,
                    record.flexible_stderr,
                    record.strict_stderr,
                    record.duration,
                    record.log_path,
                    record.generation_records_path,
                ]
            )


def _write_leaderboard_md(
    leaderboard: list[RunRecord], all_records: list[RunRecord], output_path: Path
) -> None:
    completed_count = sum(record.status == "completed" for record in all_records)
    failed_count = sum(record.status == "failed" for record in all_records)
    running_count = sum(record.status == "running" for record in all_records)

    lines = [
        "# Entropy Priority Grid Search Leaderboard",
        "",
        f"- Total runs discovered: `{len(all_records)}`",
        f"- Completed: `{completed_count}`",
        f"- Failed: `{failed_count}`",
        f"- Running: `{running_count}`",
        "",
    ]

    if leaderboard:
        best = leaderboard[0]
        lines.extend(
            [
                "## Best Run",
                "",
                "| Run | GPU | Credit Rate | Warmup | Active End | End | Top K | Flexible EM | Strict EM | Duration |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                "| {run_id} | {gpu} | {cr:.2f} | {wu:.2f} | {ae:.2f} | {ee:.2f} | {top_k} | {flex:.3f} | {strict:.3f} | {duration} |".format(
                    run_id=best.run_id,
                    gpu=best.assigned_gpu if best.assigned_gpu is not None else "",
                    cr=best.credit_rate or 0.0,
                    wu=best.warmup_ratio or 0.0,
                    ae=best.active_end_ratio or 0.0,
                    ee=best.end_ratio or 0.0,
                    top_k=best.entropy_top_k if best.entropy_top_k is not None else "",
                    flex=best.flexible_exact_match or 0.0,
                    strict=best.strict_exact_match or 0.0,
                    duration=best.duration or "",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Ranking",
            "",
            "| Rank | Run | GPU | Credit Rate | Warmup | Active End | End | Flexible EM | Strict EM | Duration | Log |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for rank, record in enumerate(leaderboard, start=1):
        lines.append(
            "| {rank} | {run_id} | {gpu} | {cr:.2f} | {wu:.2f} | {ae:.2f} | {ee:.2f} | {flex:.3f} | {strict:.3f} | {duration} | {log_path} |".format(
                rank=rank,
                run_id=record.run_id,
                gpu=record.assigned_gpu if record.assigned_gpu is not None else "",
                cr=record.credit_rate or 0.0,
                wu=record.warmup_ratio or 0.0,
                ae=record.active_end_ratio or 0.0,
                ee=record.end_ratio or 0.0,
                flex=record.flexible_exact_match or 0.0,
                strict=record.strict_exact_match or 0.0,
                duration=record.duration or "",
                log_path=record.log_path,
            )
        )

    if not leaderboard:
        lines.extend(["", "No completed runs with parsed metrics yet."])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failures_csv(records: list[RunRecord], output_path: Path) -> None:
    failed = [record for record in records if record.status == "failed"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "slug",
                "assigned_gpu",
                "exit_code",
                "note",
                "eval_log_path",
                "meta_path",
            ]
        )
        for record in failed:
            writer.writerow(
                [
                    record.run_id,
                    record.slug,
                    record.assigned_gpu,
                    record.exit_code,
                    record.note,
                    record.log_path,
                    record.meta_path,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update entropy grid manifest and leaderboard outputs."
    )
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Absolute path to the sweep root directory.",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    runs_root = sweep_root / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs_root}")

    records = _scan_records(sweep_root)
    leaderboard = _sort_leaderboard(records)

    _write_grid_manifest(records, sweep_root / "grid_manifest.csv")
    _write_leaderboard_csv(leaderboard, sweep_root / "leaderboard.csv")
    _write_leaderboard_md(leaderboard, records, sweep_root / "leaderboard.md")
    _write_failures_csv(records, sweep_root / "failures.csv")

    print(f"Updated sweep reports under: {sweep_root}")


if __name__ == "__main__":
    main()
