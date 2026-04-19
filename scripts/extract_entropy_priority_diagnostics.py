"""
Extract compact entropy-priority diagnostics from large eval JSONL files.

Run on the server from /disk/wangzhe/dllm:
  source ~/.zshrc
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ~/miniconda3/envs/dllm
  cd /disk/wangzhe/dllm
  export PYTHONPATH=.:$PYTHONPATH
  python /disk/wangzhe/dllm/scripts/extract_entropy_priority_diagnostics.py \
    --inputs /disk/wangzhe/dllm/.logs/gsm8k-*.jsonl \
    --output-prefix /disk/wangzhe/dllm/.logs/gsm8k-entropy-diagnostics
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaseRecord:
    source_file: str
    index: int | None
    question: str
    predicted_final_answer: str
    entropy_priority_effective: bool
    entropy_trigger_count: int
    entropy_selected_token_count: int
    tentative_enter_count: int
    tentative_finalize_count: int
    tentative_rollback_count: int
    baseline_finalize_count: int
    token_event_count: int
    token_events: list[dict] | None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value) -> bool:
    return bool(value)


def _read_case_records(path: Path, include_token_events: bool) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            diagnostics = item.get("sampler_diagnostics") or {}
            token_events = diagnostics.get("token_events") or []
            records.append(
                CaseRecord(
                    source_file=path.name,
                    index=item.get("index"),
                    question=str(item.get("question", "")),
                    predicted_final_answer=str(item.get("predicted_final_answer", "")),
                    entropy_priority_effective=_safe_bool(
                        diagnostics.get("entropy_priority_effective", False)
                    ),
                    entropy_trigger_count=_safe_int(
                        diagnostics.get("entropy_trigger_count", 0)
                    ),
                    entropy_selected_token_count=_safe_int(
                        diagnostics.get("entropy_selected_token_count", 0)
                    ),
                    tentative_enter_count=_safe_int(
                        diagnostics.get("tentative_enter_count", 0)
                    ),
                    tentative_finalize_count=_safe_int(
                        diagnostics.get("tentative_finalize_count", 0)
                    ),
                    tentative_rollback_count=_safe_int(
                        diagnostics.get("tentative_rollback_count", 0)
                    ),
                    baseline_finalize_count=_safe_int(
                        diagnostics.get("baseline_finalize_count", 0)
                    ),
                    token_event_count=len(token_events),
                    token_events=token_events if include_token_events else None,
                )
            )
    return records


def _write_compact_jsonl(records: list[CaseRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            item = {
                "source_file": record.source_file,
                "index": record.index,
                "question": record.question,
                "predicted_final_answer": record.predicted_final_answer,
                "entropy_priority_effective": record.entropy_priority_effective,
                "entropy_trigger_count": record.entropy_trigger_count,
                "entropy_selected_token_count": record.entropy_selected_token_count,
                "tentative_enter_count": record.tentative_enter_count,
                "tentative_finalize_count": record.tentative_finalize_count,
                "tentative_rollback_count": record.tentative_rollback_count,
                "baseline_finalize_count": record.baseline_finalize_count,
                "token_event_count": record.token_event_count,
            }
            if record.token_events is not None:
                item["token_events"] = record.token_events
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_csv(records: list[CaseRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "index",
                "entropy_priority_effective",
                "entropy_trigger_count",
                "entropy_selected_token_count",
                "tentative_enter_count",
                "tentative_finalize_count",
                "tentative_rollback_count",
                "baseline_finalize_count",
                "token_event_count",
                "question",
                "predicted_final_answer",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.source_file,
                    record.index,
                    record.entropy_priority_effective,
                    record.entropy_trigger_count,
                    record.entropy_selected_token_count,
                    record.tentative_enter_count,
                    record.tentative_finalize_count,
                    record.tentative_rollback_count,
                    record.baseline_finalize_count,
                    record.token_event_count,
                    record.question,
                    record.predicted_final_answer,
                ]
            )


def _build_summary(records: list[CaseRecord]) -> str:
    grouped: dict[str, list[CaseRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_file, []).append(record)

    lines = [
        "# Entropy Priority Diagnostics Summary",
        "",
        "| Source File | Cases | Effective Cases | Effective Ratio | Avg Trigger Count | Avg Selected Tokens | Avg Tentative Enter | Avg Tentative Finalize | Avg Tentative Rollback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for source_file, source_records in sorted(grouped.items()):
        total = len(source_records)
        effective = sum(r.entropy_priority_effective for r in source_records)
        avg_trigger = sum(r.entropy_trigger_count for r in source_records) / total
        avg_selected = (
            sum(r.entropy_selected_token_count for r in source_records) / total
        )
        avg_enter = sum(r.tentative_enter_count for r in source_records) / total
        avg_finalize = (
            sum(r.tentative_finalize_count for r in source_records) / total
        )
        avg_rollback = (
            sum(r.tentative_rollback_count for r in source_records) / total
        )
        lines.append(
            "| {source} | {total} | {effective} | {ratio:.3f} | {avg_trigger:.3f} | {avg_selected:.3f} | {avg_enter:.3f} | {avg_finalize:.3f} | {avg_rollback:.3f} |".format(
                source=source_file,
                total=total,
                effective=effective,
                ratio=effective / total if total else 0.0,
                avg_trigger=avg_trigger,
                avg_selected=avg_selected,
                avg_enter=avg_enter,
                avg_finalize=avg_finalize,
                avg_rollback=avg_rollback,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Effective Cases` counts samples with `entropy_priority_effective=True`.",
            "- `Avg Trigger Count` is the mean number of entropy trigger events per sample.",
            "- `Avg Selected Tokens` is the mean number of tokens selected by entropy priority per sample.",
            "- Tentative statistics are per-sample averages, useful for judging whether the entropy path only triggers or also survives to finalize.",
        ]
    )
    return "\n".join(lines) + "\n"


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(Path(path) for path in matched)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    unique_paths: list[Path] = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact entropy-priority diagnostics from eval JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files or glob patterns.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files, e.g. /disk/wangzhe/dllm/.logs/gsm8k-entropy-diagnostics",
    )
    parser.add_argument(
        "--include-token-events",
        action="store_true",
        help="Keep token-level event lists in the compact JSONL output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No input JSONL files matched.")

    records: list[CaseRecord] = []
    for path in input_paths:
        records.extend(_read_case_records(path, include_token_events=args.include_token_events))

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    compact_jsonl = output_prefix.with_suffix(".jsonl")
    compact_csv = output_prefix.with_suffix(".csv")
    summary_md = output_prefix.with_suffix(".md")

    _write_compact_jsonl(records, compact_jsonl)
    _write_csv(records, compact_csv)
    summary_md.write_text(_build_summary(records), encoding="utf-8")

    print(f"Wrote compact JSONL: {compact_jsonl}")
    print(f"Wrote compact CSV:   {compact_csv}")
    print(f"Wrote summary MD:    {summary_md}")


if __name__ == "__main__":
    main()
