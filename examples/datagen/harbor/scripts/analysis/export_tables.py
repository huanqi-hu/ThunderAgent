#!/usr/bin/env python3
"""Export markdown comparison tables from metrics-{default,tr}-{48..128}.json files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BATCH_SIZES = [48, 64, 80, 96, 112, 128]
ROUTERS = ["default", "tr"]

# Metric definitions for per-BS detail tables
# (display_name, json_key, higher_is_better, format_spec)
DETAIL_METRICS: list[tuple[str, str, bool, str]] = [
    ("Tokens/sec", "tokens_per_sec", True, ".1f"),
    ("Steps/min", "steps_per_min", True, ".2f"),
    ("Trajectories/min", "traj_per_min", True, ".2f"),
    ("Avg KV Hit Rate", "avg_kv_hit_rate", True, ".4f"),
    ("Weighted KV Hit Rate", "weighted_kv_hit_rate", True, ".4f"),
    ("Avg Step Latency (s)", "avg_step_latency_s", False, ".2f"),
    ("Avg Trajectory Latency (s)", "avg_traj_latency_s", False, ".1f"),
    ("Avg Evicted Tokens/Req", "avg_evicted_tokens_per_request", False, ",.0f"),
    ("KV Cache Occupancy (median)", "kv_cache_occupancy_median", False, ".4f"),
    ("Re-prefill Ratio", "prefill_compute_ratio", False, ".4f"),
    ("Avg Pause (s)", "avg_pause_s", None, ".3f"),  # None = TR-only
]

# Summary table definitions: (title, json_key, format_spec)
SUMMARY_TABLES: list[tuple[str, str, str, str]] = [
    ("Decode Throughput (tokens/sec)", "tokens_per_sec", "Tokens/sec", ".1f"),
    ("Steps/min", "steps_per_min", "Steps/min", ".2f"),
    ("Trajectories/min", "traj_per_min", "Traj/min", ".2f"),
]


def _load_all_metrics(results_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load metrics-{router}-{bs}.json into {router: {bs: data}}."""
    all_data: dict[str, dict[int, dict[str, Any]]] = {}
    for router in ROUTERS:
        all_data[router] = {}
        for bs in BATCH_SIZES:
            path = results_dir / f"metrics-{router}-{bs}.json"
            if not path.exists():
                print(f"WARNING: {path} not found, skipping", file=sys.stderr)
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            all_data[router][bs] = data
    return all_data


def _fmt(value: float | None, spec: str) -> str:
    if value is None:
        return "N/A"
    return format(value, spec)


def _comparison_str(
    dv: float | None, tv: float | None, higher_is_better: bool | None
) -> str:
    """Format comparison column: ratio with arrow or 'TR only'."""
    if higher_is_better is None:
        return "TR only"
    if dv is None or tv is None or dv == 0 or tv == 0:
        return "N/A"
    if higher_is_better:
        ratio = tv / dv
        arrow = "up" if ratio >= 1.0 else "down"
    else:
        ratio = dv / tv
        arrow = "down" if ratio >= 1.0 else "up"
    symbol = "\u2191" if arrow == "up" else "\u2193"
    return f"{ratio:.2f}x {symbol}"


def _generate_summary_tables(
    all_data: dict[str, dict[int, dict[str, Any]]]
) -> str:
    """Generate 3 throughput summary tables."""
    lines: list[str] = []
    for title, key, _col_name, fmt_spec in SUMMARY_TABLES:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Rollout BS | SGLang (default) | ThunderAgent (tr) | Improvement |")
        lines.append("|-----------|-----------------|-------------------|-------------|")
        for bs in BATCH_SIZES:
            dv = all_data.get("default", {}).get(bs, {}).get(key)
            tv = all_data.get("tr", {}).get(bs, {}).get(key)
            d_str = _fmt(dv, fmt_spec) if dv is not None else "N/A"
            t_str = _fmt(tv, fmt_spec) if tv is not None else "N/A"
            if dv is not None and tv is not None and dv > 0:
                ratio = tv / dv
                imp = f"{ratio:.2f}x"
            else:
                imp = "N/A"
            lines.append(f"| {bs:<9} | {d_str:<15} | {t_str:<17} | {imp:<11} |")
        lines.append("")
    return "\n".join(lines)


def _generate_detail_tables(
    all_data: dict[str, dict[int, dict[str, Any]]]
) -> str:
    """Generate 6 per-BS detail tables."""
    lines: list[str] = []
    for bs in BATCH_SIZES:
        d_data = all_data.get("default", {}).get(bs, {})
        t_data = all_data.get("tr", {}).get(bs, {})
        if not d_data and not t_data:
            continue

        lines.append(f"## BS={bs} Detailed Comparison")
        lines.append("")
        lines.append("| Metric | SGLang (default) | ThunderAgent (tr) | Comparison |")
        lines.append("|--------|-----------------|-------------------|------------|")

        for display_name, key, higher_is_better, fmt_spec in DETAIL_METRICS:
            dv = d_data.get(key)
            tv = t_data.get(key)
            d_str = _fmt(dv, fmt_spec) if dv is not None else "N/A"
            t_str = _fmt(tv, fmt_spec) if tv is not None else "N/A"
            comp = _comparison_str(dv, tv, higher_is_better)
            lines.append(f"| {display_name} | {d_str} | {t_str} | {comp} |")

        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export markdown comparison tables from metrics JSONs")
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing metrics-{default,tr}-{48..128}.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: results-dir/comparison_tables.md)",
    )
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    output_path: Path = args.output or (results_dir / "comparison_tables.md")

    all_data = _load_all_metrics(results_dir)

    summary = _generate_summary_tables(all_data)
    detail = _generate_detail_tables(all_data)

    full_output = f"# TR vs Default Router Comparison\n\n{summary}\n{detail}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_output, encoding="utf-8")
    print(full_output)
    print(f"\nWritten to: {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
