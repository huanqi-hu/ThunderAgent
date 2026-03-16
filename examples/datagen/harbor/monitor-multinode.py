#!/usr/bin/env python3
"""Live experiment monitor for multi-node SGLang + ThunderAgent experiments.

Reads metrics from the multi-node run directory layout:
    runs/<name>/
        thunderagent_profiles/step_profiles.csv
        sglang-<nodeA>/sglang_metrics.csv
        sglang-<nodeB>/sglang_metrics.csv

Usage:
    python3 monitor-multinode.py                                  # auto-detect active runs
    python3 monitor-multinode.py tr-128-multinode                 # specific run
    python3 monitor-multinode.py --align tr-128-multinode default-128-multinode
    python3 monitor-multinode.py --minutes 60 tr-128-multinode default-128-multinode
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

RUNS_ROOT = Path(__file__).parent / "runs"

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _get_thunderagent_start(run_dir: Path) -> float | None:
    """Return the earliest completed_at from step_profiles, or None."""
    prof_csv = run_dir / "thunderagent_profiles" / "step_profiles.csv"
    if not prof_csv.exists():
        return None
    t_min = float("inf")
    with open(prof_csv) as f:
        for row in csv.DictReader(f):
            t = safe_float(row.get("completed_at", 0))
            if t > 0:
                t_min = min(t_min, t)
    return t_min if t_min < float("inf") else None


def _find_sglang_csv_files(run_dir: Path) -> list[Path]:
    """Find all sglang_metrics.csv files in a multi-node run directory."""
    csvs = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sglang-"):
            csv_path = d / "sglang_metrics.csv"
            if csv_path.exists():
                csvs.append(csv_path)
    # Fallback: single-node layout
    if not csvs:
        for candidate in [run_dir / "metrics" / "sglang_metrics.csv",
                          run_dir / "sglang_metrics.csv"]:
            if candidate.exists():
                csvs.append(candidate)
                break
    return csvs


def _load_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_sglang_metrics(run_dir: Path, max_duration_s: float | None = None) -> dict:
    """Compute aggregate SGLang server metrics across all backends."""
    csv_files = _find_sglang_csv_files(run_dir)
    if not csv_files:
        return {}

    ta_start = _get_thunderagent_start(run_dir)

    # Load all backend CSVs
    per_backend = {}
    for csv_path in csv_files:
        backend_name = csv_path.parent.name
        rows = _load_csv_rows(csv_path)
        if len(rows) < 2:
            continue
        per_backend[backend_name] = rows

    if not per_backend:
        return {}

    # For each backend, filter to active period and compute metrics
    backend_metrics = []
    for name, rows in per_backend.items():
        if ta_start is not None:
            t_active_start = ta_start
        else:
            active_rows = [r for r in rows
                           if safe_float(r.get("gen_throughput_total", 0)) > 10]
            if active_rows:
                t_active_start = safe_float(active_rows[0]["timestamp"])
            else:
                t_active_start = safe_float(rows[0]["timestamp"])

        if max_duration_s is not None:
            t_cutoff = t_active_start + max_duration_s
            filtered = [r for r in rows
                        if t_active_start <= safe_float(r["timestamp"]) <= t_cutoff]
        else:
            filtered = [r for r in rows
                        if safe_float(r["timestamp"]) >= t_active_start]

        if len(filtered) < 2:
            continue

        t_end = safe_float(filtered[-1]["timestamp"])
        active_duration = t_end - t_active_start

        active_rows = [r for r in filtered if safe_float(r.get("gen_throughput_total", 0)) > 0]
        throughputs = [safe_float(r.get("gen_throughput_total", 0)) for r in active_rows] if active_rows else [0]
        running_reqs = [safe_float(r.get("num_running_reqs", 0)) for r in filtered]
        waiting_reqs = [safe_float(r.get("num_waiting_reqs", 0)) for r in filtered]
        token_usage = [safe_float(r.get("token_usage", 0)) for r in filtered]
        cache_hit = [safe_float(r.get("cache_hit_rate", 0)) for r in filtered]

        evict_start = safe_float(filtered[0].get("evicted_tokens_total", 0))
        evict_end = safe_float(filtered[-1].get("evicted_tokens_total", 0))
        pc_start = safe_float(filtered[0].get("prefill_cache_tokens_total", 0))
        pc_end = safe_float(filtered[-1].get("prefill_cache_tokens_total", 0))
        pcomp_start = safe_float(filtered[0].get("prefill_compute_tokens_total", 0))
        pcomp_end = safe_float(filtered[-1].get("prefill_compute_tokens_total", 0))

        backend_metrics.append({
            "name": name,
            "active_duration_s": active_duration,
            "t_active_start": t_active_start,
            "n_samples": len(filtered),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "max_throughput": max(throughputs) if throughputs else 0,
            "avg_running": sum(running_reqs) / len(running_reqs),
            "max_running": max(running_reqs),
            "avg_waiting": sum(waiting_reqs) / len(waiting_reqs),
            "avg_token_usage": sum(token_usage) / len(token_usage),
            "avg_cache_hit": sum(cache_hit) / len(cache_hit),
            "evicted_tokens": evict_end - evict_start,
            "prefill_cache_tokens": pc_end - pc_start,
            "prefill_compute_tokens": pcomp_end - pcomp_start,
        })

    if not backend_metrics:
        return {}

    # Aggregate across backends
    total_evict = sum(b["evicted_tokens"] for b in backend_metrics)
    total_cache = sum(b["prefill_cache_tokens"] for b in backend_metrics)
    total_compute = sum(b["prefill_compute_tokens"] for b in backend_metrics)
    total_prefill = total_cache + total_compute

    return {
        "n_backends": len(backend_metrics),
        "active_duration_s": max(b["active_duration_s"] for b in backend_metrics),
        "t_active_start": min(b["t_active_start"] for b in backend_metrics),
        "avg_throughput": sum(b["avg_throughput"] for b in backend_metrics),
        "max_throughput": sum(b["max_throughput"] for b in backend_metrics),
        "avg_running": sum(b["avg_running"] for b in backend_metrics),
        "max_running": sum(b["max_running"] for b in backend_metrics),
        "avg_waiting": sum(b["avg_waiting"] for b in backend_metrics),
        "avg_token_usage": sum(b["avg_token_usage"] for b in backend_metrics) / len(backend_metrics),
        "avg_cache_hit": sum(b["avg_cache_hit"] for b in backend_metrics) / len(backend_metrics),
        "evicted_tokens": total_evict,
        "prefill_cache_tokens": total_cache,
        "prefill_compute_tokens": total_compute,
        "server_cache_ratio": total_cache / total_prefill if total_prefill > 0 else 0,
        "per_backend": backend_metrics,
    }


def load_step_profiles(run_dir: Path, max_duration_s: float | None = None) -> dict:
    """Compute aggregate ThunderAgent step profile metrics."""
    csv_path = run_dir / "thunderagent_profiles" / "step_profiles.csv"
    if not csv_path.exists():
        return {}

    all_steps = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            all_steps.append(row)

    if not all_steps:
        return {}

    completed_times = [safe_float(s.get("completed_at", 0)) for s in all_steps
                       if safe_float(s.get("completed_at", 0)) > 0]
    if len(completed_times) < 2:
        return {"n_steps": len(all_steps), "n_programs": 0}

    t_start = min(completed_times)

    if max_duration_s is not None:
        t_cutoff = t_start + max_duration_s
        steps = [s for s in all_steps
                 if safe_float(s.get("completed_at", 0)) <= t_cutoff
                 and safe_float(s.get("completed_at", 0)) > 0]
    else:
        steps = [s for s in all_steps if safe_float(s.get("completed_at", 0)) > 0]

    if not steps:
        return {"n_steps": 0, "n_programs": 0}

    programs = defaultdict(list)
    for s in steps:
        programs[s["program_id"]].append(s)

    t_end = max(safe_float(s["completed_at"]) for s in steps)
    wall_s = t_end - t_start

    steps_per_sec = len(steps) / wall_s if wall_s > 0 else 0

    prompt_tokens = [safe_float(s.get("prompt_tokens", 0)) for s in steps]
    completion_tokens = [safe_float(s.get("completion_tokens", 0)) for s in steps]
    cached_tokens = [safe_float(s.get("cached_tokens", 0)) for s in steps if s.get("cached_tokens")]
    kv_hits = [safe_float(s.get("kv_hit_rate", 0)) for s in steps if s.get("kv_hit_rate")]
    pause_times = [safe_float(s.get("pause_s", 0)) for s in steps]

    total_prompt = sum(prompt_tokens)
    total_cached = sum(cached_tokens) if cached_tokens else 0
    total_completion = sum(completion_tokens)

    effective_tok_s = total_completion / wall_s if wall_s > 0 else 0

    max_steps = {pid: max(int(s.get("step_id", 0)) for s in ss) for pid, ss in programs.items()}
    trials_done = sum(1 for ms in max_steps.values() if ms >= 2)

    paused_steps = [p for p in pause_times if p > 0.01]

    return {
        "n_steps": len(steps),
        "n_programs": len(programs),
        "trials_started": len(programs),
        "trials_multi_step": trials_done,
        "wall_s": wall_s,
        "t_start": t_start,
        "steps_per_sec": steps_per_sec,
        "effective_tok_s": effective_tok_s,
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0,
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "total_completion_tokens": total_completion,
        "agent_cache_ratio": total_cached / total_prompt if total_prompt > 0 else 0,
        "avg_kv_hit": sum(kv_hits) / len(kv_hits) if kv_hits else 0,
        "avg_pause_s": sum(pause_times) / len(pause_times) if pause_times else 0,
        "paused_step_count": len(paused_steps),
        "paused_step_pct": len(paused_steps) / len(steps) * 100 if steps else 0,
        "avg_pause_when_paused": sum(paused_steps) / len(paused_steps) if paused_steps else 0,
    }


# ─── Formatting helpers ──────────────────────────────────────────────

def fmt_num(v, decimals=1):
    if v >= 1e9:
        return f"{v/1e9:.{decimals}f}B"
    if v >= 1e6:
        return f"{v/1e6:.{decimals}f}M"
    if v >= 1e3:
        return f"{v/1e3:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    return f"{v*100:.1f}%" if v <= 1 else f"{v:.1f}%"


def fmt_duration(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"


# ─── Display ─────────────────────────────────────────────────────────

def print_experiment(name: str, sglang: dict, prof: dict):
    """Print a formatted summary for one experiment."""
    # Parse router from name (e.g. "tr-128-multinode" -> "tr")
    parts = name.split("-")
    router = parts[0] if parts else "unknown"
    color = RED if router == "tr" else CYAN

    duration = sglang.get("active_duration_s", prof.get("wall_s", 0))
    dur_str = fmt_duration(duration) if duration > 0 else "N/A"
    n_backends = sglang.get("n_backends", "?")

    print(f"\n{BOLD}{color}{'='*64}")
    print(f"  {name.upper()}  (router={router}, {n_backends} backends)  [{dur_str}]")
    print(f"{'='*64}{RESET}")

    if prof:
        print(f"  {BOLD}Agent Throughput{RESET}")
        print(f"    Steps:           {prof['n_steps']:>8,}    ({prof.get('steps_per_sec',0):.2f}/s)")
        print(f"    Programs:        {prof['n_programs']:>8,}    (multi-step: {prof.get('trials_multi_step',0)})")
        print(f"    Eff. tok/s:      {prof.get('effective_tok_s',0):>8.0f}")
        print(f"    Avg completion:  {prof.get('avg_completion_tokens',0):>8.0f} tokens/step")

    if sglang:
        print(f"  {BOLD}SGLang Server (combined {n_backends} backends){RESET}")
        print(f"    Total throughput:{sglang['avg_throughput']:>8.0f} tok/s  (max: {sglang['max_throughput']:.0f})")
        print(f"    Total running:   {sglang['avg_running']:>8.1f}    (max: {sglang['max_running']:.0f})")
        print(f"    Avg KV usage:    {fmt_pct(sglang['avg_token_usage']):>8}  (mean across backends)")
        print(f"    Avg cache hit:   {fmt_pct(sglang['avg_cache_hit']):>8}  (server-side instantaneous)")

        # Per-backend breakdown
        per_backend = sglang.get("per_backend", [])
        if len(per_backend) > 1:
            print(f"  {BOLD}Per-Backend Breakdown{RESET}")
            for b in per_backend:
                print(f"    {b['name']:<20} {b['avg_throughput']:>6.0f} tok/s  "
                      f"running={b['avg_running']:.0f}  KV={fmt_pct(b['avg_token_usage'])}")

    if sglang and sglang.get("evicted_tokens", 0) > 0:
        print(f"  {BOLD}Eviction & Prefill (total across backends){RESET}")
        print(f"    Evicted tokens:  {fmt_num(sglang['evicted_tokens']):>8}")
        print(f"    Prefill cache:   {fmt_num(sglang['prefill_cache_tokens']):>8}")
        print(f"    Prefill compute: {fmt_num(sglang['prefill_compute_tokens']):>8}")
        print(f"    Server hit ratio:{fmt_pct(sglang['server_cache_ratio']):>8}  (cumul. cache/(cache+compute))")

    if prof and prof.get("total_prompt_tokens", 0) > 0:
        print(f"  {BOLD}Agent Cache (from step_profiles){RESET}")
        print(f"    Total prompt:    {fmt_num(prof['total_prompt_tokens']):>8}")
        print(f"    Total cached:    {fmt_num(prof['total_cached_tokens']):>8}")
        print(f"    Agent hit ratio: {fmt_pct(prof['agent_cache_ratio']):>8}")
        print(f"    Avg kv_hit_rate: {fmt_pct(prof['avg_kv_hit']):>8}  (per-step mean)")

    if prof and router == "tr":
        print(f"  {BOLD}Pause Scheduling (tr router){RESET}")
        print(f"    Avg pause/step:  {prof.get('avg_pause_s',0):>8.3f}s")
        print(f"    Steps paused:    {prof.get('paused_step_count',0):>8,}  ({prof.get('paused_step_pct',0):.1f}%)")
        if prof.get("paused_step_count", 0) > 0:
            print(f"    Avg when paused: {prof.get('avg_pause_when_paused',0):>8.1f}s")


def print_comparison_table(results: list[tuple[str, dict, dict]]):
    """Print a compact comparison table across experiments."""
    print(f"\n{BOLD}{'='*106}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*106}{RESET}")

    header = (f"{'Experiment':<24} {'Duration':>8} {'Steps':>7} {'Steps/s':>7} "
              f"{'Eff tok/s':>9} {'Sum gen':>8} {'Cache hit':>9} {'Evicted':>9} {'Programs':>8}")
    print(f"  {DIM}{header}{RESET}")
    print(f"  {'-'*102}")

    for name, sglang, prof in results:
        parts = name.split("-")
        router = parts[0] if parts else "unknown"
        color = RED if router == "tr" else CYAN
        dur = sglang.get("active_duration_s", prof.get("wall_s", 0))

        steps = prof.get("n_steps", 0)
        sps = prof.get("steps_per_sec", 0)
        eff = prof.get("effective_tok_s", 0)
        gen = sglang.get("avg_throughput", 0)
        hit = prof.get("avg_kv_hit", 0)
        evict = sglang.get("evicted_tokens", 0)
        progs = prof.get("n_programs", 0)

        row = (
            f"{color}{name:<24}{RESET} "
            f"{fmt_duration(dur):>8} "
            f"{steps:>7,} "
            f"{sps:>7.2f} "
            f"{eff:>9.0f} "
            f"{gen:>8.0f} "
            f"{fmt_pct(hit):>9} "
            f"{fmt_num(evict):>9} "
            f"{progs:>8,}"
        )
        print(f"  {row}")

    print()


def _get_active_duration(run_dir: Path) -> float:
    """Quick pass to determine how long an experiment has been active."""
    prof_csv = run_dir / "thunderagent_profiles" / "step_profiles.csv"
    if prof_csv.exists():
        times = []
        with open(prof_csv) as f:
            for row in csv.DictReader(f):
                t = safe_float(row.get("completed_at", 0))
                if t > 0:
                    times.append(t)
        if len(times) >= 2:
            return max(times) - min(times)

    # Fallback: sglang metrics
    csv_files = _find_sglang_csv_files(run_dir)
    for csv_path in csv_files:
        rows = _load_csv_rows(csv_path)
        active = [r for r in rows if safe_float(r.get("gen_throughput_total", 0)) > 0]
        if active:
            return safe_float(active[-1]["timestamp"]) - safe_float(active[0]["timestamp"])
    return 0.0


def _compute_align_cutoffs(selected: list[str]) -> dict[str, float]:
    """For each pair with matching BS, compute min active duration."""
    by_bs: dict[str, dict[str, str]] = defaultdict(dict)
    for name in selected:
        # Parse: "tr-128-multinode" -> router="tr", bs_key="128-multinode"
        parts = name.split("-", 1)
        if len(parts) == 2:
            router = parts[0]
            bs_key = parts[1]
            by_bs[bs_key][router] = name

    cutoffs: dict[str, float] = {}
    for bs_key, pair in by_bs.items():
        if "default" not in pair or "tr" not in pair:
            continue
        dur_default = _get_active_duration(RUNS_ROOT / pair["default"])
        dur_tr = _get_active_duration(RUNS_ROOT / pair["tr"])
        if dur_default <= 0 or dur_tr <= 0:
            continue
        min_dur = min(dur_default, dur_tr)
        cutoffs[pair["default"]] = min_dur
        cutoffs[pair["tr"]] = min_dur
    return cutoffs


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node experiment monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 monitor-multinode.py                          # auto-detect\n"
               "  python3 monitor-multinode.py tr-128-multinode          # specific run\n"
               "  python3 monitor-multinode.py --align tr-128 default-128\n"
               "  python3 monitor-multinode.py --minutes 120 tr-128 default-128\n",
    )
    parser.add_argument("experiments", nargs="*",
                        help="Run directory names under runs/. Omit to auto-detect.")
    parser.add_argument("--align", action="store_true",
                        help="Align each default/tr pair to min(duration)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="Override alignment window in minutes (implies --align)")
    args = parser.parse_args()

    if args.minutes is not None:
        args.align = True

    # Determine which runs to show
    if args.experiments:
        selected = args.experiments
    else:
        # Auto-detect: any run dir with CSV data
        selected = []
        if RUNS_ROOT.exists():
            for d in sorted(RUNS_ROOT.iterdir()):
                if d.is_dir() and any(d.rglob("*.csv")):
                    selected.append(d.name)

    if not selected:
        print("No experiments found. Check RUNS_ROOT:", RUNS_ROOT)
        sys.exit(1)

    # Compute per-experiment time cutoffs
    cutoffs: dict[str, float | None] = {n: None for n in selected}
    if args.align:
        if args.minutes is not None:
            cutoffs = {n: args.minutes * 60 for n in selected}
        else:
            cutoffs.update(_compute_align_cutoffs(selected))

    # Header
    print(f"{BOLD}Multi-Node Experiment Monitor{RESET}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.align:
        aligned = {n: c for n, c in cutoffs.items() if c is not None}
        for name, window in aligned.items():
            print(f"{YELLOW}  [align] {name}: first {fmt_duration(window)}{RESET}")

    # Load and display
    results = []
    for name in selected:
        run_dir = RUNS_ROOT / name
        max_dur = cutoffs.get(name)
        sglang = load_sglang_metrics(run_dir, max_duration_s=max_dur)
        prof = load_step_profiles(run_dir, max_duration_s=max_dur)
        print_experiment(name, sglang, prof)
        results.append((name, sglang, prof))

    if len(results) > 1:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
