"""``harbor jobs status`` — live status of a running or completed job.

Works for both local and distributed runs by scanning the job directory
for result.json files and worker heartbeats.

Usage::

    harbor jobs status jobs-swebench/my-run
    harbor jobs status jobs-swebench/my-run --watch    # auto-refresh
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

from typer import Argument, Option


def status(
    job_dir: Annotated[
        Path,
        Argument(help="Path to the job directory"),
    ],
    watch: Annotated[
        bool,
        Option("--watch", "-w", help="Auto-refresh every 5 seconds"),
    ] = False,
) -> None:
    """Show status of a running or completed job."""
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    console = Console()

    if not job_dir.exists():
        console.print(f"[red]Job directory not found: {job_dir}[/red]")
        raise SystemExit(1)

    def render() -> Table:
        grid = Table.grid(padding=(0, 2))
        grid.add_column()

        # Scan results.
        solved = total = errors = in_progress = 0
        exc_types: dict[str, int] = {}
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir() or trial_dir.name.startswith("."):
                continue
            rj = trial_dir / "result.json"
            if not rj.exists():
                continue
            try:
                r = json.loads(rj.read_text())
                # Skip preliminary results (trial still running).
                if r.get("finished_at") is None:
                    in_progress += 1
                    continue
                total += 1
                vr = r.get("verifier_result") or {}
                rw = (vr.get("rewards") or {}).get("reward", 0)
                if rw and rw > 0:
                    solved += 1
                exc = r.get("exception_info")
                if exc and isinstance(exc, dict):
                    errors += 1
                    et = exc.get("exception_type", "?")
                    exc_types[et] = exc_types.get(et, 0) + 1
            except Exception:
                continue

        rate = f"{solved / total * 100:.1f}%" if total else "—"

        # Summary table.
        summary = Table(title=f"Job: {job_dir.name}", show_lines=True)
        summary.add_column("Metric", style="cyan", width=20)
        summary.add_column("Value", width=50)
        summary.add_row("Completed", str(total))
        if in_progress:
            summary.add_row(
                "In progress",
                f"[yellow]{in_progress}[/yellow]",
            )
        summary.add_row(
            "Solved",
            f"[green]{solved}[/green]/{total} ({rate})" if total else "—",
        )
        summary.add_row(
            "Errors",
            f"[red]{errors}[/red]" if errors else "[green]0[/green]",
        )
        if exc_types:
            exc_str = ", ".join(
                f"{et}: {c}" for et, c in sorted(exc_types.items(), key=lambda x: -x[1])
            )
            summary.add_row("Error types", exc_str)

        # Queue status (distributed).
        queue_dir = job_dir / ".queue"
        if queue_dir.exists():
            pending = sum(1 for f in queue_dir.iterdir() if f.name.endswith(".json"))
            summary.add_row("Queue pending", str(pending))

        grid.add_row(summary)

        # Worker heartbeats (distributed).
        workers_dir = job_dir / ".workers"
        if workers_dir.exists():
            heartbeats = []
            for f in sorted(workers_dir.iterdir()):
                if f.suffix != ".json":
                    continue
                try:
                    hb = json.loads(f.read_text())
                    heartbeats.append(hb)
                except Exception:
                    continue

            if heartbeats:
                wt = Table(title="Workers", show_lines=True)
                wt.add_column("Node", style="cyan", width=25)
                wt.add_column("Status", width=10)
                wt.add_column("Claimed", width=8, justify="right")
                wt.add_column("Done", width=8, justify="right")
                wt.add_column("Failed", width=8, justify="right")
                wt.add_column("Last Active", width=20)

                for hb in heartbeats:
                    status_style = {
                        "running": "green",
                        "starting": "yellow",
                        "draining": "yellow",
                        "stopped": "dim",
                    }.get(hb.get("status", ""), "white")
                    last = hb.get("last_active", "")
                    if last:
                        # Show relative time.
                        try:
                            from datetime import datetime, timezone

                            dt = datetime.fromisoformat(last)
                            ago = (datetime.now(timezone.utc) - dt).total_seconds()
                            if ago < 60:
                                last = f"{ago:.0f}s ago"
                            elif ago < 3600:
                                last = f"{ago / 60:.0f}m ago"
                            else:
                                last = f"{ago / 3600:.1f}h ago"
                        except Exception:
                            pass
                    wt.add_row(
                        hb.get("node_name", "?"),
                        f"[{status_style}]{hb.get('status', '?')}[/]",
                        str(hb.get("trials_claimed", 0)),
                        str(hb.get("trials_completed", 0)),
                        str(hb.get("trials_failed", 0)),
                        last,
                    )
                grid.add_row(wt)

        return grid

    if watch:
        with Live(render(), refresh_per_second=0.2, console=console) as live:
            try:
                while True:
                    time.sleep(5)
                    live.update(render())
            except KeyboardInterrupt:
                pass
    else:
        console.print(render())
