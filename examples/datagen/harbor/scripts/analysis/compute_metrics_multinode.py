#!/usr/bin/env python3
"""Compute metrics for a multi-node experiment run and output JSON.

Extends compute_metrics.py to handle multiple SGLang backends. Each backend's
sglang_metrics.csv lives in a separate subdirectory (e.g., sglang-nodeA/,
sglang-nodeB/). Server-side metrics (evictions, prefill cache/compute, KV
occupancy) are summed across all backends.

Usage:
    python compute_metrics_multinode.py --run-dir runs/tr-128-multinode/
    python compute_metrics_multinode.py --run-dir runs/default-128-multinode/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_run_dir_name(run_dir: Path) -> tuple[str, int]:
    """Extract router and bs from run-dir basename like 'tr-128-multinode'."""
    name = run_dir.name
    # Strip known suffixes
    for suffix in ("-multinode",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Split on the last hyphen
    parts = name.rsplit("-", 1)
    if len(parts) != 2:
        print(f"WARNING: cannot parse router/bs from '{run_dir.name}', using defaults", file=sys.stderr)
        return "unknown", 0
    router, bs_str = parts
    try:
        bs = int(bs_str)
    except ValueError:
        print(f"WARNING: cannot parse bs from '{bs_str}', using 0", file=sys.stderr)
        bs = 0
    return router, bs


def find_sglang_metrics_dirs(run_dir: Path) -> list[Path]:
    """Find all sglang-<node>/ subdirectories containing sglang_metrics.csv."""
    dirs = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sglang-"):
            csv = d / "sglang_metrics.csv"
            if csv.is_file():
                dirs.append(d)
    # Fallback: single-node layout (metrics in root or metrics/ subdir)
    if not dirs:
        for candidate in [run_dir / "sglang_metrics.csv", run_dir / "metrics" / "sglang_metrics.csv"]:
            if candidate.is_file():
                dirs.append(candidate.parent)
                break
    return dirs


def load_step_profiles(run_dir: Path) -> pd.DataFrame | None:
    """Load step_profiles.csv from thunderagent_profiles/."""
    path = run_dir / "thunderagent_profiles" / "step_profiles.csv"
    if not path.is_file():
        print(f"WARNING: step_profiles.csv not found at {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if df.empty:
        print("WARNING: step_profiles.csv is empty", file=sys.stderr)
        return None
    return df


def load_and_merge_sglang_metrics(dirs: list[Path]) -> pd.DataFrame | None:
    """Load and merge sglang_metrics.csv from multiple backend directories.

    For each timestamp bucket (5s intervals), sums counters across backends
    and averages gauges.
    """
    frames = []
    for d in dirs:
        csv_path = d / "sglang_metrics.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if not df.empty:
            df["_backend"] = d.name
            frames.append(df)

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0].drop(columns=["_backend"], errors="ignore")

    # Merge: bucket by timestamp (round to nearest 5s) and aggregate
    all_df = pd.concat(frames, ignore_index=True)
    all_df["ts_bucket"] = (all_df["timestamp"] / 5).round() * 5

    # Counter columns: sum across backends
    counter_cols = [
        "evicted_tokens_total",
        "prefill_cache_tokens_total",
        "prefill_compute_tokens_total",
        "retracted_requests_total",
        "retracted_input_tokens_total",
        "retracted_output_tokens_total",
    ]

    # Gauge columns: sum (gen_throughput, running_reqs, waiting_reqs) or average (token_usage, cache_hit_rate)
    sum_gauge_cols = ["gen_throughput_total", "num_running_reqs", "num_waiting_reqs", "num_queue_reqs"]
    avg_gauge_cols = ["token_usage", "cache_hit_rate"]

    agg_dict = {"timestamp": "mean"}
    for col in counter_cols:
        if col in all_df.columns:
            agg_dict[col] = "sum"
    for col in sum_gauge_cols:
        if col in all_df.columns:
            agg_dict[col] = "sum"
    for col in avg_gauge_cols:
        if col in all_df.columns:
            agg_dict[col] = "mean"
    # Also include retracted_reqs if present
    if "retracted_reqs" in all_df.columns:
        agg_dict["retracted_reqs"] = "sum"

    merged = all_df.groupby("ts_bucket", as_index=False).agg(agg_dict)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def compute_throughput_metrics(df: pd.DataFrame, elapsed_seconds: float) -> dict:
    """Compute throughput metrics from step_profiles."""
    metrics: dict = {}
    elapsed_minutes = elapsed_seconds / 60.0

    if "completion_tokens" in df.columns:
        total_completion = df["completion_tokens"].sum()
        metrics["tokens_per_sec"] = round(total_completion / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0
    else:
        metrics["tokens_per_sec"] = None

    total_steps = len(df)
    metrics["steps_per_min"] = round(total_steps / elapsed_minutes, 1) if elapsed_minutes > 0 else 0.0

    if "program_id" in df.columns:
        total_programs = df["program_id"].nunique()
        metrics["traj_per_min"] = round(total_programs / elapsed_minutes, 2) if elapsed_minutes > 0 else 0.0
    else:
        metrics["traj_per_min"] = None

    return metrics


def compute_cache_metrics(df: pd.DataFrame) -> dict:
    """Compute cache efficiency metrics from step_profiles."""
    metrics: dict = {}

    if "kv_hit_rate" not in df.columns:
        return {"avg_kv_hit_rate": None, "weighted_kv_hit_rate": None, "kv_hit_rate_by_step": {}}

    valid = df.dropna(subset=["kv_hit_rate"])

    if len(valid) > 0:
        metrics["avg_kv_hit_rate"] = round(float(valid["kv_hit_rate"].mean()), 4)
    else:
        metrics["avg_kv_hit_rate"] = None

    if len(valid) > 0 and "prompt_tokens" in df.columns:
        prompt_tokens = valid["prompt_tokens"].fillna(0)
        total_prompt = prompt_tokens.sum()
        if total_prompt > 0:
            weighted = (valid["kv_hit_rate"] * prompt_tokens).sum() / total_prompt
            metrics["weighted_kv_hit_rate"] = round(float(weighted), 4)
        else:
            metrics["weighted_kv_hit_rate"] = None
    else:
        metrics["weighted_kv_hit_rate"] = None

    if "step_id" in df.columns and len(valid) > 0:
        grouped = valid.groupby("step_id")["kv_hit_rate"]
        counts = grouped.count()
        means = grouped.mean()
        by_step = {}
        for step_id in counts.index:
            if counts[step_id] >= 10:
                by_step[str(step_id)] = round(float(means[step_id]), 4)
        metrics["kv_hit_rate_by_step"] = by_step
    else:
        metrics["kv_hit_rate_by_step"] = {}

    return metrics


def compute_latency_metrics(df: pd.DataFrame) -> dict:
    """Compute latency metrics from step_profiles."""
    metrics: dict = {}

    latency_cols = ["prefill_s", "decode_s", "pause_s"]
    available_cols = [c for c in latency_cols if c in df.columns]
    if available_cols:
        step_latency = df[available_cols].fillna(0).sum(axis=1)
        metrics["avg_step_latency_s"] = round(float(step_latency.mean()), 2)
    else:
        metrics["avg_step_latency_s"] = None

    if "program_id" in df.columns and "completed_at" in df.columns:
        valid = df.dropna(subset=["completed_at"])
        if len(valid) > 0:
            grouped = valid.groupby("program_id")["completed_at"]
            traj_latencies = grouped.max() - grouped.min()
            traj_latencies = traj_latencies[traj_latencies > 0]
            if len(traj_latencies) > 0:
                metrics["avg_traj_latency_s"] = round(float(traj_latencies.mean()), 1)
            else:
                metrics["avg_traj_latency_s"] = None
        else:
            metrics["avg_traj_latency_s"] = None
    else:
        metrics["avg_traj_latency_s"] = None

    return metrics


def compute_server_metrics(sglang_df: pd.DataFrame, total_steps: int) -> dict:
    """Compute server-side metrics from merged sglang_metrics.csv."""
    metrics: dict = {}

    if "evicted_tokens_total" in sglang_df.columns and total_steps > 0:
        col = sglang_df["evicted_tokens_total"].dropna()
        if len(col) >= 2:
            delta = float(col.iloc[-1]) - float(col.iloc[0])
            metrics["avg_evicted_tokens_per_request"] = round(delta / total_steps, 1)
        else:
            metrics["avg_evicted_tokens_per_request"] = None
    else:
        metrics["avg_evicted_tokens_per_request"] = None

    if "token_usage" in sglang_df.columns and "timestamp" in sglang_df.columns:
        ts = sglang_df["timestamp"].dropna()
        if len(ts) > 0:
            t0 = ts.iloc[0]
            mask = sglang_df["timestamp"] >= (t0 + 60)
            filtered = sglang_df.loc[mask, "token_usage"].dropna()
            if len(filtered) > 0:
                metrics["kv_cache_occupancy_median"] = round(float(filtered.median()), 4)
            else:
                metrics["kv_cache_occupancy_median"] = None
        else:
            metrics["kv_cache_occupancy_median"] = None
    else:
        metrics["kv_cache_occupancy_median"] = None

    compute_col = "prefill_compute_tokens_total"
    cache_col = "prefill_cache_tokens_total"
    if compute_col in sglang_df.columns and cache_col in sglang_df.columns:
        compute_vals = sglang_df[compute_col].dropna()
        cache_vals = sglang_df[cache_col].dropna()
        if len(compute_vals) >= 2 and len(cache_vals) >= 2:
            delta_compute = float(compute_vals.iloc[-1]) - float(compute_vals.iloc[0])
            delta_cache = float(cache_vals.iloc[-1]) - float(cache_vals.iloc[0])
            denom = delta_compute + delta_cache
            if denom > 0:
                metrics["prefill_compute_ratio"] = round(delta_compute / denom, 4)
            else:
                metrics["prefill_compute_ratio"] = None
        else:
            metrics["prefill_compute_ratio"] = None
    else:
        metrics["prefill_compute_ratio"] = None

    return metrics


def compute_pause_metrics(df: pd.DataFrame) -> dict:
    """Compute ThunderAgent pause metrics (TR-specific)."""
    metrics: dict = {}

    if "pause_s" not in df.columns:
        return {"pct_steps_paused": None, "avg_pause_per_step_s": None}

    total_steps = len(df)
    pause_vals = df["pause_s"].fillna(0)

    if total_steps > 0:
        paused_count = int((pause_vals > 0.01).sum())
        metrics["pct_steps_paused"] = round(paused_count / total_steps * 100, 1)
    else:
        metrics["pct_steps_paused"] = None

    metrics["avg_pause_per_step_s"] = round(float(pause_vals.mean()), 4)

    return metrics


def compute_all_metrics(run_dir: Path) -> dict:
    """Compute all metrics for one multi-node experiment run."""
    router, bs = parse_run_dir_name(run_dir)

    step_df = load_step_profiles(run_dir)
    sglang_dirs = find_sglang_metrics_dirs(run_dir)
    sglang_df = load_and_merge_sglang_metrics(sglang_dirs)

    n_backends = len(sglang_dirs)

    if step_df is None:
        print("ERROR: cannot compute metrics without step_profiles.csv", file=sys.stderr)
        return {
            "bs": bs,
            "router": router,
            "n_backends": n_backends,
            "error": "step_profiles.csv not found or empty",
        }

    # Elapsed time
    if "completed_at" in step_df.columns:
        completed = step_df["completed_at"].dropna()
        if len(completed) >= 2:
            elapsed_seconds = float(completed.max() - completed.min())
        else:
            elapsed_seconds = 0.0
    else:
        elapsed_seconds = 0.0

    total_steps = len(step_df)
    total_programs = step_df["program_id"].nunique() if "program_id" in step_df.columns else 0

    throughput = compute_throughput_metrics(step_df, elapsed_seconds)
    cache = compute_cache_metrics(step_df)
    latency = compute_latency_metrics(step_df)
    pause = compute_pause_metrics(step_df)

    if sglang_df is not None:
        server = compute_server_metrics(sglang_df, total_steps)
    else:
        server = {
            "avg_evicted_tokens_per_request": None,
            "kv_cache_occupancy_median": None,
            "prefill_compute_ratio": None,
        }

    result = {
        "bs": bs,
        "router": router,
        "n_backends": n_backends,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "total_steps": total_steps,
        "total_programs": total_programs,
        "tokens_per_sec": throughput["tokens_per_sec"],
        "steps_per_min": throughput["steps_per_min"],
        "traj_per_min": throughput["traj_per_min"],
        "avg_kv_hit_rate": cache["avg_kv_hit_rate"],
        "weighted_kv_hit_rate": cache["weighted_kv_hit_rate"],
        "avg_step_latency_s": latency["avg_step_latency_s"],
        "avg_traj_latency_s": latency["avg_traj_latency_s"],
        "avg_evicted_tokens_per_request": server["avg_evicted_tokens_per_request"],
        "kv_cache_occupancy_median": server["kv_cache_occupancy_median"],
        "prefill_compute_ratio": server["prefill_compute_ratio"],
        "pct_steps_paused": pause["pct_steps_paused"],
        "avg_pause_per_step_s": pause["avg_pause_per_step_s"],
        "kv_hit_rate_by_step": cache["kv_hit_rate_by_step"],
    }

    return result


def print_summary(result: dict) -> None:
    """Print a human-readable summary to stderr."""
    print("=" * 60, file=sys.stderr)
    print(f"  Run: {result['router']}-{result['bs']} ({result.get('n_backends', '?')} backends)", file=sys.stderr)
    print(f"  Elapsed: {result['elapsed_seconds']:.0f}s ({result['elapsed_seconds'] / 60:.1f} min)", file=sys.stderr)
    print(f"  Steps: {result['total_steps']}, Programs: {result['total_programs']}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    def fmt(val, suffix="", precision=2):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{precision}f}{suffix}"
        return f"{val}{suffix}"

    print(f"  Throughput:", file=sys.stderr)
    print(f"    tokens/sec:    {fmt(result.get('tokens_per_sec'), precision=1)}", file=sys.stderr)
    print(f"    steps/min:     {fmt(result.get('steps_per_min'), precision=1)}", file=sys.stderr)
    print(f"    traj/min:      {fmt(result.get('traj_per_min'))}", file=sys.stderr)

    print(f"  Cache:", file=sys.stderr)
    print(f"    avg hit rate:  {fmt(result.get('avg_kv_hit_rate'), precision=4)}", file=sys.stderr)
    print(f"    weighted hit:  {fmt(result.get('weighted_kv_hit_rate'), precision=4)}", file=sys.stderr)

    print(f"  Latency:", file=sys.stderr)
    print(f"    avg step:      {fmt(result.get('avg_step_latency_s'), 's')}", file=sys.stderr)
    print(f"    avg traj:      {fmt(result.get('avg_traj_latency_s'), 's', precision=1)}", file=sys.stderr)

    print(f"  Server (aggregated across {result.get('n_backends', '?')} backends):", file=sys.stderr)
    print(f"    evict/req:     {fmt(result.get('avg_evicted_tokens_per_request'), precision=1)}", file=sys.stderr)
    print(f"    kv occupancy:  {fmt(result.get('kv_cache_occupancy_median'), precision=4)}", file=sys.stderr)
    print(f"    prefill comp:  {fmt(result.get('prefill_compute_ratio'), precision=4)}", file=sys.stderr)

    print(f"  Pause:", file=sys.stderr)
    print(f"    pct paused:    {fmt(result.get('pct_steps_paused'), '%', precision=1)}", file=sys.stderr)
    print(f"    avg pause:     {fmt(result.get('avg_pause_per_step_s'), 's', precision=4)}", file=sys.stderr)

    by_step = result.get("kv_hit_rate_by_step", {})
    if by_step:
        print(f"  KV Hit Rate by Step ({len(by_step)} steps):", file=sys.stderr)
        for sid in sorted(by_step.keys(), key=lambda x: int(x)):
            print(f"    step {sid}: {by_step[sid]:.4f}", file=sys.stderr)

    print("=" * 60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for a multi-node experiment run and output JSON."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to runs/{router}-{bs}-multinode/ directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: auto-generated in run dir)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    result = compute_all_metrics(run_dir)
    print_summary(result)

    output_json = json.dumps(result, indent=2)

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n")
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        auto_path = run_dir / "metrics.json"
        auto_path.write_text(output_json + "\n")
        print(f"Wrote {auto_path}", file=sys.stderr)

    print(output_json)


if __name__ == "__main__":
    main()
