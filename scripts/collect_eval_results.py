"""
Collect llada eval metrics from multiple log files into one comparison report.

Run from repo root on the server:
  source ~/.zshrc
  source /home/wangzhe/miniconda3/etc/profile.d/conda.sh
  conda activate dllm
  cd /disk/wangzhe/dllm
  python /disk/wangzhe/dllm/scripts/collect_eval_results.py \
    --logs /disk/wangzhe/dllm/.logs/gsm8k-*.log \
    --output /disk/wangzhe/dllm/.logs/gsm8k-comparison.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(
    r"llada \((?P<config>.+?)\), gen_kwargs: \(.+?\), limit: (?P<limit>[0-9.]+), "
    r"num_fewshot: (?P<num_fewshot>\d+), batch_size: (?P<batch_size>\d+)"
)
GENERATING_RE = re.compile(
    r"Generating\.\.\.:\s+100%\|.+\[(?P<duration>[0-9:]+)<00:00,"
)
FLOAT_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")


@dataclass
class EvalResult:
    log_path: str
    variant: str
    limit: int | None
    num_fewshot: int | None
    batch_size: int | None
    steps: str
    block_size: str
    flexible_exact_match: float | None
    flexible_stderr: float | None
    strict_exact_match: float | None
    strict_stderr: float | None
    duration: str | None
    comparable: bool = True
    note: str = ""


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _parse_markdown_row(line: str) -> list[str] | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None

    columns = [part.strip() for part in line.split("|")[1:-1]]
    if not columns:
        return None

    # Skip separator rows like |---|---:|---|
    if all(column and set(column) <= {"-", ":"} for column in columns):
        return None

    return columns


def _extract_metric_row(columns: list[str]) -> tuple[str, float, float] | None:
    if not columns:
        return None

    first_column = columns[0].strip()
    if first_column not in {"gsm8k_cot", ""}:
        return None

    filter_name = None
    for column in columns:
        normalized = column.strip()
        if normalized in {"flexible-extract", "strict-match"}:
            filter_name = normalized
            break
    if filter_name is None:
        return None

    numeric_values = [
        float(column) for column in columns if FLOAT_RE.match(column.strip())
    ]
    if len(numeric_values) < 2:
        return None

    value = numeric_values[-2]
    stderr = numeric_values[-1]
    return filter_name, value, stderr


def parse_model_args(config_blob: str) -> dict[str, str]:
    pieces: list[str] = []
    current = []
    bracket_depth = 0
    for char in config_blob:
        if char == "," and bracket_depth == 0:
            pieces.append("".join(current).strip())
            current = []
            continue
        current.append(char)
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
    if current:
        pieces.append("".join(current).strip())

    parsed: dict[str, str] = {}
    for piece in pieces:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def infer_variant(config: dict[str, str], log_name: str) -> str:
    if config.get("enable_structure_priority") == "True":
        return "structure_priority"
    if config.get("enable_tentative_commit") == "True":
        return "tentative_remask"
    if config.get("enable_entropy_priority") == "True":
        return "entropy_only"
    if "baseline" in log_name:
        return "baseline"
    return "unknown"


def parse_eval_log(path: Path) -> EvalResult:
    summary_match = None
    duration = None
    flexible_exact_match = None
    flexible_stderr = None
    strict_exact_match = None
    strict_stderr = None
    variant = "unknown"
    steps = "?"
    block_size = "?"
    num_fewshot = None
    batch_size = None
    limit = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = _strip_ansi(raw_line).strip()
            if not summary_match:
                summary_match = SUMMARY_RE.search(line)
                if summary_match:
                    config = parse_model_args(summary_match.group("config"))
                    variant = infer_variant(config, path.name)
                    steps = config.get("steps", "?")
                    block_size = config.get("block_size", "?")
                    limit = int(float(summary_match.group("limit")))
                    num_fewshot = int(summary_match.group("num_fewshot"))
                    batch_size = int(summary_match.group("batch_size"))

            gen_match = GENERATING_RE.search(line)
            if gen_match:
                duration = gen_match.group("duration")

            row = _parse_markdown_row(line)
            metric_row = _extract_metric_row(row) if row is not None else None
            if metric_row is None:
                continue

            filter_name, value, stderr = metric_row
            if filter_name == "flexible-extract":
                flexible_exact_match = value
                flexible_stderr = stderr
            elif filter_name == "strict-match":
                strict_exact_match = value
                strict_stderr = stderr

    return EvalResult(
        log_path=str(path),
        variant=variant,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        steps=steps,
        block_size=block_size,
        flexible_exact_match=flexible_exact_match,
        flexible_stderr=flexible_stderr,
        strict_exact_match=strict_exact_match,
        strict_stderr=strict_stderr,
        duration=duration,
    )


def build_report(results: list[EvalResult]) -> str:
    if not results:
        return "# Eval Comparison\n\nNo results found.\n"

    baseline = next((result for result in results if result.variant == "baseline"), None)
    common_limit = baseline.limit if baseline else None

    for result in results:
        if common_limit is not None and result.limit != common_limit:
            result.comparable = False
            result.note = (
                f"limit={result.limit} differs from baseline limit={common_limit}; "
                "do not compare directly."
            )

    lines = [
        "# GSM8K Comparison",
        "",
        "| Variant | Limit | Steps | Block | Flexible EM | Strict EM | Flexible Δ vs Base | Strict Δ vs Base | Stderr (flex/strict) | Duration | Comparable | Log | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]

    for result in results:
        flex_delta = ""
        strict_delta = ""
        if baseline and result.flexible_exact_match is not None and baseline.flexible_exact_match is not None:
            flex_delta = f"{result.flexible_exact_match - baseline.flexible_exact_match:+.3f}"
        if baseline and result.strict_exact_match is not None and baseline.strict_exact_match is not None:
            strict_delta = f"{result.strict_exact_match - baseline.strict_exact_match:+.3f}"

        stderr = ""
        if result.flexible_stderr is not None and result.strict_stderr is not None:
            stderr = f"{result.flexible_stderr:.4f} / {result.strict_stderr:.4f}"

        lines.append(
            "| {variant} | {limit} | {steps} | {block} | {flex} | {strict} | {flex_delta} | {strict_delta} | {stderr} | {duration} | {comparable} | {log_path} | {note} |".format(
                variant=result.variant,
                limit=result.limit if result.limit is not None else "",
                steps=result.steps,
                block=result.block_size,
                flex=f"{result.flexible_exact_match:.3f}" if result.flexible_exact_match is not None else "",
                strict=f"{result.strict_exact_match:.3f}" if result.strict_exact_match is not None else "",
                flex_delta=flex_delta,
                strict_delta=strict_delta,
                stderr=stderr,
                duration=result.duration or "",
                comparable="yes" if result.comparable else "no",
                log_path=result.log_path,
                note=result.note,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Flexible EM` corresponds to `flexible-extract`.",
            "- `Strict EM` corresponds to `strict-match`.",
            "- Rows with a different `limit` from baseline are marked as not directly comparable.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(results: list[EvalResult], output_path: Path) -> Path:
    csv_path = output_path.with_suffix(".csv")
    baseline = next((result for result in results if result.variant == "baseline"), None)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "limit",
                "steps",
                "block_size",
                "flexible_exact_match",
                "strict_exact_match",
                "flexible_stderr",
                "strict_stderr",
                "flexible_delta_vs_baseline",
                "strict_delta_vs_baseline",
                "duration",
                "comparable",
                "note",
                "log_path",
            ]
        )
        for result in results:
            flex_delta = ""
            strict_delta = ""
            if baseline and result.flexible_exact_match is not None and baseline.flexible_exact_match is not None:
                flex_delta = f"{result.flexible_exact_match - baseline.flexible_exact_match:.6f}"
            if baseline and result.strict_exact_match is not None and baseline.strict_exact_match is not None:
                strict_delta = f"{result.strict_exact_match - baseline.strict_exact_match:.6f}"
            writer.writerow(
                [
                    result.variant,
                    result.limit,
                    result.steps,
                    result.block_size,
                    result.flexible_exact_match,
                    result.strict_exact_match,
                    result.flexible_stderr,
                    result.strict_stderr,
                    flex_delta,
                    strict_delta,
                    result.duration,
                    result.comparable,
                    result.note,
                    result.log_path,
                ]
            )

    return csv_path


def resolve_logs(patterns: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(Path(pattern))
    deduped = sorted({path.resolve() for path in resolved if path.exists()})
    return [Path(path) for path in deduped]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect llada eval results from logs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Log paths or glob patterns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Markdown output path.",
    )
    args = parser.parse_args()

    log_paths = resolve_logs(args.logs)
    if not log_paths:
        raise FileNotFoundError("No matching log files found.")

    results = [parse_eval_log(path) for path in log_paths]
    order = {
        "baseline": 0,
        "entropy_only": 1,
        "tentative_remask": 2,
        "structure_priority": 3,
        "unknown": 9,
    }
    results.sort(key=lambda item: (order.get(item.variant, 9), item.log_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(results), encoding="utf-8")
    csv_path = write_csv(results, output_path)

    print(f"Wrote markdown report to: {output_path}")
    print(f"Wrote csv summary to: {csv_path}")


if __name__ == "__main__":
    main()
