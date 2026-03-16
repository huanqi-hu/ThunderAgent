"""``harbor jobs kill`` — stop a running job and clean up all resources.

Handles both local and distributed jobs: kills workers, removes Docker
containers, and reports status.

Usage::

    harbor jobs kill jobs/my-run
    harbor jobs kill jobs/my-run --force  # skip confirmation
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from typer import Argument, Option

if TYPE_CHECKING:
    from rich.console import Console


def kill(
    job_dir: Annotated[
        Path,
        Argument(help="Path to the job directory"),
    ],
    force: Annotated[
        bool,
        Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Kill a running job and clean up all resources."""
    from rich.console import Console

    console = Console()

    if not job_dir.exists():
        console.print(f"[red]Job directory not found: {job_dir}[/red]")
        raise SystemExit(1)

    # Detect if distributed (has .workers/ dir with heartbeats)
    workers_dir = job_dir / ".workers"
    is_distributed = workers_dir.exists() and any(
        f.suffix == ".json" for f in workers_dir.iterdir()
    )

    if is_distributed:
        nodes = _get_nodes(workers_dir)
        if not force:
            console.print(
                f"Kill distributed job [bold]{job_dir.name}[/bold] "
                f"across {len(nodes)} nodes?"
            )
            for n in nodes:
                console.print(f"  {n['node_name']} (pid={n['pid']})")
            if input("Proceed? [y/N] ").lower() != "y":
                return
        asyncio.run(_kill_distributed(nodes, console))
    else:
        if not force:
            console.print(f"Kill local job [bold]{job_dir.name}[/bold]?")
            if input("Proceed? [y/N] ").lower() != "y":
                return
        _kill_local(console)


def _get_nodes(workers_dir: Path) -> list[dict]:
    """Read worker heartbeats to find active nodes."""
    nodes = []
    for f in sorted(workers_dir.iterdir()):
        if f.suffix != ".json":
            continue
        try:
            hb = json.loads(f.read_text())
            if hb.get("status") not in ("stopped",):
                nodes.append(hb)
        except Exception:
            continue
    return nodes


async def _kill_distributed(nodes: list[dict], console: "Console") -> None:
    """Kill workers and clean containers on all nodes."""
    import getpass

    user = getpass.getuser()

    async def kill_node(hb: dict) -> str:
        hostname = hb["node_name"]
        pid = hb.get("pid")

        # Kill worker process
        kill_cmd = f"kill {pid} 2>/dev/null; sleep 2; kill -9 {pid} 2>/dev/null"

        # Clean containers
        clean_cmd = (
            "docker ps -q --filter name=__ | xargs -r docker rm -f 2>/dev/null | wc -l"
        )

        cmd = f"{kill_cmd}; CLEANED=$({clean_cmd}); echo $CLEANED"

        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                "ConnectTimeout=10",
                f"{user}@{hostname}",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            cleaned = stdout.decode().strip()
            return f"  [green]{hostname}[/green]: killed pid {pid}, removed {cleaned} containers"
        except Exception as e:
            return f"  [red]{hostname}[/red]: failed ({e})"

    console.print("Killing workers and cleaning containers...")
    results = await asyncio.gather(*[kill_node(hb) for hb in nodes])
    for r in results:
        console.print(r)
    console.print("[bold]Done.[/bold]")


def _kill_local(console: "Console") -> None:
    """Kill local harbor run process and clean containers."""
    import subprocess

    # Kill harbor run
    result = subprocess.run(
        ["pkill", "-f", "harbor run"],
        capture_output=True,
    )
    if result.returncode == 0:
        console.print("  Killed harbor run process")
    else:
        console.print("  No harbor run process found")

    # Clean containers
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=__"],
        capture_output=True,
        text=True,
    )
    ids = result.stdout.strip().split()
    if ids and ids[0]:
        subprocess.run(
            ["docker", "rm", "-f"] + ids,
            capture_output=True,
        )
        console.print(f"  Removed {len(ids)} containers")
    else:
        console.print("  No containers to clean")

    console.print("[bold]Done.[/bold]")
