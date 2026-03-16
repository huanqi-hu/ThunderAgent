"""``harbor warmup`` — pre-pull and pre-build Docker images for a dataset.

Pre-pulling images on a node before starting an eval ensures the trial
pipeline doesn't waste LLM slots on image I/O.

Usage::

    # Pull base images + build overlays locally
    harbor warmup pull -d swebench-verified

    # Pull on remote nodes via SSH (for distributed runs)
    harbor warmup pull -d swebench-verified --nodes coder-008,coder-015,coder-016

    # Just check what's cached
    harbor warmup pull -d swebench-verified --dry-run
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from typer import Option, Typer

warmup_app = Typer(no_args_is_help=True)


@warmup_app.command(name="pull")
def warmup(
    dataset_name_version: Annotated[
        str | None,
        Option("--dataset", "-d", help="Dataset name@version"),
    ] = None,
    path: Annotated[
        Path | None,
        Option("--path", "-p", help="Path to a local dataset directory"),
    ] = None,
    n_concurrent: Annotated[
        int,
        Option("--n-concurrent", "-n", help="Max concurrent pulls/builds"),
    ] = 32,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Registry URL for remote dataset"),
    ] = None,
    nodes: Annotated[
        list[str] | None,
        Option(
            "--nodes",
            help="Run warmup on remote nodes via SSH (comma-separated or repeated)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Option("--dry-run", help="Just report cache status, don't pull/build"),
    ] = False,
) -> None:
    """Pre-pull all Docker images for a dataset.

    Handles both prebuilt images (docker_image in task.toml) and
    Dockerfile-based tasks (pulls base image + builds overlay).
    """
    asyncio.run(
        _warmup(dataset_name_version, path, n_concurrent, registry_url, nodes, dry_run)
    )


async def _warmup(
    dataset_name_version: str | None,
    path: Path | None,
    n_concurrent: int,
    registry_url: str | None,
    nodes: list[str] | None,
    dry_run: bool,
) -> None:
    # If nodes specified, SSH to each and run warmup there.
    if nodes:
        await _warmup_remote(
            dataset_name_version, path, n_concurrent, registry_url, nodes
        )
        return

    from harbor.models.job.config import (
        LocalDatasetConfig,
        RegistryDatasetConfig,
    )
    from harbor.models.registry import RemoteRegistryInfo
    from harbor.models.task.task import Task

    # Resolve task configs.
    if path:
        ds = LocalDatasetConfig(path=path)
        task_configs = ds.get_task_configs(disable_verification=True)
    elif dataset_name_version:
        parts = dataset_name_version.split("@", 1)
        name = parts[0]
        version = parts[1] if len(parts) > 1 else None
        registry = (
            RemoteRegistryInfo(url=registry_url)
            if registry_url
            else RemoteRegistryInfo()
        )
        ds = RegistryDatasetConfig(registry=registry, name=name, version=version)
        task_configs = ds.get_task_configs(disable_verification=True)
    else:
        raise ValueError("Provide --dataset or --path")

    # Download git-based tasks so we can inspect their Dockerfiles.
    from harbor.tasks.client import TaskClient

    git_configs = [tc for tc in task_configs if tc.is_git_task()]
    if git_configs:
        client = TaskClient()
        task_ids = [tc.get_task_id() for tc in git_configs]
        print(f"Downloading {len(task_ids)} task definitions...")
        resolved_paths = client.download_tasks(task_ids=task_ids, overwrite=False)
    else:
        resolved_paths = [tc.path for tc in task_configs if tc.path]

    # Collect images: prebuilt (docker_image) and Dockerfile-based (base + overlay).
    prebuilt: dict[str, str] = {}  # image -> task_name
    base_images: dict[str, str] = {}  # base_image -> task_name
    overlays: dict[str, Path] = {}  # overlay_name -> build_context_dir

    for task_path in resolved_paths:
        try:
            task = Task(task_dir=task_path)
            img = task.config.environment.docker_image
            if img:
                if img not in prebuilt:
                    prebuilt[img] = task.name
            else:
                # Dockerfile-based: extract base image and overlay name.
                dockerfile = task.paths.environment_dir / "Dockerfile"
                if dockerfile.exists():
                    overlay_name = f"hb__{task.name}"
                    if overlay_name not in overlays:
                        overlays[overlay_name] = task.paths.environment_dir
                    # Extract FROM base image.
                    for line in dockerfile.read_text().splitlines():
                        stripped = line.strip()
                        if stripped.upper().startswith("FROM "):
                            base_img = stripped.split()[1]
                            if base_img not in base_images:
                                base_images[base_img] = task.name
                            break
        except Exception:
            continue

    all_images = {**prebuilt, **base_images}
    print(
        f"Dataset: {len(task_configs)} tasks, "
        f"{len(prebuilt)} prebuilt, {len(base_images)} base, "
        f"{len(overlays)} overlays"
    )

    # Check cache status.
    cached_count = 0
    to_pull: list[str] = []
    for img in all_images:
        if await _image_exists(img):
            cached_count += 1
        else:
            to_pull.append(img)

    overlay_cached = 0
    to_build: dict[str, Path] = {}
    for name, ctx in overlays.items():
        if await _image_exists(name):
            overlay_cached += 1
        else:
            to_build[name] = ctx

    print(
        f"Cache: {cached_count}/{len(all_images)} images cached, "
        f"{overlay_cached}/{len(overlays)} overlays cached"
    )
    print(f"Need: {len(to_pull)} pulls, {len(to_build)} builds")

    if dry_run:
        return

    if not to_pull and not to_build:
        print("All images cached. Node is warm.")
        return

    # Pull missing images.
    if to_pull:
        await _pull_images(to_pull, n_concurrent)

    # Build missing overlays.
    if to_build:
        await _build_overlays(to_build, n_concurrent)

    print("Warmup complete.")


async def _image_exists(image: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "image",
        "inspect",
        image,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0


async def _pull_images(images: list[str], n_concurrent: int) -> None:
    sem = asyncio.Semaphore(n_concurrent)
    pulled = 0
    failed = 0

    progress = Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("Pulling images...", total=len(images))

        async def pull_one(img: str) -> None:
            nonlocal pulled, failed
            async with sem:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "pull",
                    img,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                if proc.returncode == 0:
                    pulled += 1
                else:
                    failed += 1
                progress.advance(task)
                progress.update(
                    task,
                    description=f"Pulling... ({pulled} ok, {failed} failed)",
                )

        await asyncio.gather(*[pull_one(img) for img in images])

    print(f"Pull: {pulled} ok, {failed} failed")


async def _build_overlays(overlays: dict[str, Path], n_concurrent: int) -> None:
    sem = asyncio.Semaphore(n_concurrent)
    built = 0
    failed_names: list[str] = []

    progress = Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )

    async def build_one(name: str, ctx: Path, attempt: int = 0) -> bool:
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "-t",
                name,
                str(ctx),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True
            if attempt > 0:
                err = stderr.decode()[-200:].strip() if stderr else "unknown"
                print(f"  FAILED {name}: {err}")
                return False
        # Retry outside the semaphore to avoid deadlock.
        return await build_one(name, ctx, attempt=1)

    with progress:
        task = progress.add_task("Building overlays...", total=len(overlays))

        async def tracked_build(name: str, ctx: Path) -> None:
            nonlocal built
            ok = await build_one(name, ctx)
            if ok:
                built += 1
            else:
                failed_names.append(name)
            progress.advance(task)
            progress.update(
                task,
                description=f"Building... ({built} ok, {len(failed_names)} failed)",
            )

        await asyncio.gather(
            *[tracked_build(name, ctx) for name, ctx in overlays.items()]
        )

    print(f"Build: {built} ok, {len(failed_names)} failed")
    if failed_names:
        print(f"Failed overlays: {', '.join(failed_names[:10])}")
        if len(failed_names) > 10:
            print(f"  ... and {len(failed_names) - 10} more")


async def _warmup_remote(
    dataset_name_version: str | None,
    path: Path | None,
    n_concurrent: int,
    registry_url: str | None,
    nodes: list[str],
) -> None:
    """Run warmup on remote nodes via SSH with live progress."""
    import getpass
    import re
    import sys

    from rich.live import Live
    from rich.table import Table

    user = getpass.getuser()
    harbor_bin = sys.argv[0]

    # Build the remote warmup command (without --nodes).
    cmd_parts = [harbor_bin, "warmup", "pull"]
    if dataset_name_version:
        cmd_parts.extend(["-d", dataset_name_version])
    if path:
        cmd_parts.extend(["-p", str(path)])
    cmd_parts.extend(["-n", str(n_concurrent)])
    if registry_url:
        cmd_parts.extend(["--registry-url", registry_url])
    remote_cmd = " ".join(cmd_parts)

    # Per-node status tracking.
    node_status: dict[str, dict] = {
        node: {"status": "connecting", "detail": "", "progress": ""} for node in nodes
    }

    def render_table() -> Table:
        table = Table(title="Multi-Node Warmup", show_lines=True)
        table.add_column("Node", style="cyan", width=30)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=50)
        for node in nodes:
            s = node_status[node]
            style = {
                "connecting": "yellow",
                "pulling": "blue",
                "building": "blue",
                "done": "green",
                "cached": "green",
                "FAILED": "red",
            }.get(s["status"], "white")
            detail = s["detail"] or s["progress"]
            table.add_row(node, f"[{style}]{s['status']}[/]", detail)
        return table

    async def warmup_node(node: str, live: Live) -> None:
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{node}",
            remote_cmd,
        ]
        node_status[node]["status"] = "starting"
        live.update(render_table())

        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Stream output line by line for live progress.
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().strip()
            if not text:
                continue

            # Parse progress indicators from the output.
            if "Pulling" in text or "pulling" in text.lower():
                node_status[node]["status"] = "pulling"
            elif "Building" in text or "building" in text.lower():
                node_status[node]["status"] = "building"
            elif "All images cached" in text or "0 pulls, 0 builds" in text:
                node_status[node]["status"] = "cached"
            elif "complete" in text.lower():
                node_status[node]["status"] = "done"

            # Extract progress bar info (e.g., "123/500").
            match = re.search(r"(\d+)/(\d+)", text)
            if match:
                node_status[node]["progress"] = f"{match.group(1)}/{match.group(2)}"

            # Keep last meaningful detail.
            if any(
                kw in text.lower()
                for kw in ["pull", "build", "cache", "need", "dataset"]
            ):
                node_status[node]["detail"] = text[:50]

            live.update(render_table())

        await proc.wait()
        if proc.returncode != 0:
            node_status[node]["status"] = "FAILED"
        elif node_status[node]["status"] not in ("done", "cached"):
            node_status[node]["status"] = "done"
        live.update(render_table())

    with Live(render_table(), refresh_per_second=2) as live:
        await asyncio.gather(*[warmup_node(node, live) for node in nodes])

    # Final summary.
    ok = sum(1 for s in node_status.values() if s["status"] in ("done", "cached"))
    failed = len(nodes) - ok
    print(f"\nWarmup complete: {ok}/{len(nodes)} nodes ready", end="")
    if failed:
        print(f", {failed} failed")
    else:
        print()
