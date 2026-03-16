#!/usr/bin/env python3
"""Generate 11 PNG comparison plots from metrics-{default,tr}-{48..128}.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

BATCH_SIZES = [48, 64, 80, 96, 112, 128]
ROUTERS = ["default", "tr"]

# Visual style (reference: plot_hle_qwen3_8b.py, plot_throughput_2h.py)
STYLE = {
    "default": {"color": "#2ca02c", "marker": "o", "label": "SGLang (default)"},
    "tr": {"color": "#e5745b", "marker": "s", "label": "ThunderAgent (tr)"},
}
FIGSIZE = (8, 5)
DPI = 150
GRID_ALPHA = 0.3

# Color families for kv_hit_rate_by_step plot (plot 11)
_BLUE_FAMILY = ["#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b", "#041e42"]
_RED_FAMILY = ["#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
COLOR_FAMILIES_KV = {
    "default": {bs: c for bs, c in zip(BATCH_SIZES, _BLUE_FAMILY)},
    "tr": {bs: c for bs, c in zip(BATCH_SIZES, _RED_FAMILY)},
}

# Plot definitions: (filename, title, ylabel, json_key)
PLOT_DEFS: list[tuple[str, str, str, str]] = [
    ("plot_01_decode_throughput.png", "Decode Throughput", "Tokens/sec", "tokens_per_sec"),
    ("plot_02_steps_per_min.png", "Agent Steps / min", "Steps/min", "steps_per_min"),
    ("plot_03_traj_per_min.png", "Trajectories Completed / min", "Trajectories/min", "traj_per_min"),
    ("plot_04_kv_hit_rate.png", "Per-Step KV Hit Rate", "KV Hit Rate", "avg_kv_hit_rate"),
    ("plot_05_weighted_kv_hit.png", "Weighted KV Hit Rate", "Weighted KV Hit Rate", "weighted_kv_hit_rate"),
    ("plot_06_step_latency.png", "Avg Step Latency", "Latency (seconds)", "avg_step_latency_s"),
    ("plot_07_traj_latency.png", "Avg Trajectory Latency", "Latency (seconds)", "avg_traj_latency_s"),
    ("plot_08_evicted_per_req.png", "Avg Evicted Tokens per Request", "Evicted Tokens", "avg_evicted_tokens_per_request"),
    ("plot_09_kv_occupancy.png", "Median KV Cache Occupancy", "Occupancy", "kv_cache_occupancy_median"),
    ("plot_10_prefill_compute.png", "Re-prefill Ratio", "Compute Ratio", "prefill_compute_ratio"),
]


def _load_all_metrics(results_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load metrics-{router}-{bs}.json into {router: {bs: data}}."""
    all_data: dict[str, dict[int, dict[str, Any]]] = {}
    for router in ROUTERS:
        all_data[router] = {}
        for bs in BATCH_SIZES:
            path = results_dir / f"metrics-{router}-{bs}.json"
            if not path.exists():
                print(f"WARNING: {path} not found, skipping")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            all_data[router][bs] = data
    return all_data


def _extract(
    all_data: dict[str, dict[int, dict[str, Any]]], key: str
) -> dict[str, dict[int, float]]:
    """Extract a single metric key across all routers and batch sizes."""
    result: dict[str, dict[int, float]] = {}
    for router in ROUTERS:
        result[router] = {}
        for bs in BATCH_SIZES:
            m = all_data.get(router, {}).get(bs, {})
            v = m.get(key)
            if v is not None:
                result[router][bs] = float(v)
    return result


def _plot_two_line(
    data: dict[str, dict[int, float]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Create a two-line plot (default vs tr) with ratio annotations."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for router in ROUTERS:
        if router not in data:
            continue
        by_bs = data[router]
        xs = sorted(by_bs.keys())
        ys = [by_bs[c] for c in xs]
        st = STYLE[router]
        ax.plot(
            xs, ys,
            label=st["label"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Rollout batch size", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=9)
    ax.set_xticks(BATCH_SIZES)

    # Annotate TR/default ratio at each point
    if "default" in data and "tr" in data:
        for bs in BATCH_SIZES:
            if bs in data["default"] and bs in data["tr"]:
                dv = data["default"][bs]
                tv = data["tr"][bs]
                if dv > 0:
                    ratio = tv / dv
                    ax.annotate(
                        f"{ratio:.2f}x",
                        xy=(bs, tv),
                        xytext=(0, 10),
                        textcoords="offset points",
                        fontsize=8,
                        color=STYLE["tr"]["color"],
                        ha="center",
                        fontweight="bold",
                    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_kv_hit_vs_step(
    all_data: dict[str, dict[int, dict[str, Any]]], output_path: Path
) -> None:
    """Plot 11: KV hit rate vs agent step_id (12 lines)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    for router in ROUTERS:
        for bs in BATCH_SIZES:
            m = all_data.get(router, {}).get(bs, {})
            by_step = m.get("kv_hit_rate_by_step")
            if not by_step:
                continue

            # by_step is {str(step_id): float(avg_kv_hit_rate)}
            step_ids = sorted(int(k) for k in by_step.keys())
            # Clip to 1..50
            step_ids = [s for s in step_ids if 1 <= s <= 50]
            kv_vals = [by_step[str(s)] for s in step_ids]

            color = COLOR_FAMILIES_KV[router][bs]
            router_label = "SGLang (default)" if router == "default" else "ThunderAgent (tr)"
            ax.plot(
                step_ids,
                kv_vals,
                label=f"{router_label} bs={bs}",
                color=color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.85,
            )

    ax.set_xlabel("Agent Step (ReAct turn)", fontsize=11)
    ax.set_ylabel("Avg KV Hit Rate", fontsize=11)
    ax.set_title("KV Cache Hit Rate vs Agent Step", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlim(1, 50)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate comparison plots from metrics JSONs")
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing metrics-{default,tr}-{48..128}.json",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG output (default: same as results-dir)",
    )
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = _load_all_metrics(results_dir)

    # Plots 1-10: two-line metric vs BS
    for filename, title, ylabel, key in PLOT_DEFS:
        metric_data = _extract(all_data, key)
        has_data = any(metric_data.get(r) for r in ROUTERS)
        if not has_data:
            print(f"SKIP {filename}: no data for key '{key}'")
            continue
        _plot_two_line(metric_data, ylabel, title, output_dir / filename)

    # Plot 11: KV hit rate vs step_id
    _plot_kv_hit_vs_step(all_data, output_dir / "plot_11_kv_hit_vs_step.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
