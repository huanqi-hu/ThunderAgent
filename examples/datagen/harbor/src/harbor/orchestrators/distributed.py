"""Distributed orchestrator: coordinate workers across multiple nodes.

Writes trial configs to an NFS-based work queue, launches workers on
remote nodes via SSH, and monitors results.  Workers claim tasks via
atomic ``os.rename()`` (work-stealing), so fast nodes naturally do
more work.

Usage::

    harbor run --distributed \\
        --nodes coder-008,coder-015,coder-016 \\
        --n-concurrent-per-node 128 \\
        --dataset swebench-verified ...
"""

from __future__ import annotations

import asyncio
import getpass
import os
import shlex
import shutil
import signal
from pathlib import Path

from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from harbor.distributed.heartbeat import (
    is_worker_alive,
    read_all_heartbeats,
)
from harbor.distributed.queue import NFSWorkQueue
from harbor.metrics.base import BaseMetric
from harbor.models.job.config import RetryConfig
from harbor.models.orchestrator_type import OrchestratorType
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.orchestrators.base import BaseOrchestrator
from harbor.utils.logger import logger


class NodeConfig(BaseModel):
    """Configuration for a single worker node."""

    hostname: str
    n_concurrent: int | None = None
    ssh_user: str | None = None
    ssh_port: int = 22
    env_vars: dict[str, str] = Field(default_factory=dict)


class DistributedOrchestrator(BaseOrchestrator):
    """Coordinate distributed trial execution across multiple nodes."""

    POLL_INTERVAL_SEC = 5.0
    WORKER_TIMEOUT_SEC = 300

    @staticmethod
    def type() -> OrchestratorType:
        return OrchestratorType.DISTRIBUTED

    def __init__(
        self,
        trial_configs: list[TrialConfig],
        n_concurrent_trials: int,
        metrics: dict[str, list[BaseMetric]],
        quiet: bool = False,
        plain_output: bool = False,
        retry_config: RetryConfig | None = None,
        *,
        nodes: list[dict | NodeConfig] | None = None,
        job_dir: Path | str | None = None,
        worker_python: str | None = None,
        ssh_options: list[str] | None = None,
        manual_workers: bool = False,
    ):
        super().__init__(
            trial_configs=trial_configs,
            n_concurrent_trials=n_concurrent_trials,
            metrics=metrics,
            quiet=quiet,
            plain_output=plain_output,
            retry_config=retry_config,
        )
        self._logger = logger.getChild("distributed")

        if not nodes:
            raise ValueError(
                "DistributedOrchestrator requires --nodes. "
                "Use --nodes host1,host2,... to specify worker nodes."
            )

        self._nodes = [NodeConfig(**n) if isinstance(n, dict) else n for n in nodes]
        self._job_dir = Path(job_dir) if job_dir else None

        if worker_python is None:
            import sys

            worker_python = shutil.which("harbor") or sys.argv[0]
        self._worker_python = worker_python
        self._ssh_options = ssh_options or [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]
        self._current_user = getpass.getuser()
        self._manual_workers = manual_workers
        self._trial_results: list[TrialResult] = []

    async def run(self) -> list[TrialResult]:
        if not self._job_dir:
            raise ValueError("job_dir must be set for DistributedOrchestrator.")
        self._job_dir = self._job_dir.resolve()

        # Register signal handlers for graceful shutdown.
        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown.set)

        queue = NFSWorkQueue(self._job_dir)

        # Resolve trials_dir to absolute so remote workers write results
        # to the correct NFS path regardless of their cwd.
        for config in self._trial_configs:
            config.trials_dir = config.trials_dir.resolve()

        if not self._manual_workers:
            # Pre-flight: verify nodes are reachable and Docker is working.
            await self._preflight_checks()

        # Clean stale worker state from previous runs.
        self._clean_stale_worker_state()

        # Detect multi-rollout: group configs by task so all rollouts
        # of the same task are claimed by the same worker (enables
        # snapshot-based setup).
        has_multi = len(
            set(c.task.get_task_id().get_name() for c in self._trial_configs)
        ) < len(self._trial_configs)

        if has_multi:
            n_written = queue.write_task_groups(self._trial_configs)
            n_tasks = n_written
            n_rollouts = len(self._trial_configs) // n_tasks if n_tasks else 0
            self._logger.info(
                f"Multi-rollout: wrote {n_written} task groups "
                f"({n_rollouts} rollouts each, "
                f"{queue.pending_count} pending)"
            )
        else:
            n_written = queue.write_configs(self._trial_configs)
        self._logger.info(
            f"Wrote {n_written} configs to queue ({queue.pending_count} pending)"
        )

        if not self._manual_workers:
            await self._launch_workers()
        else:
            self._logger.info(
                "Manual worker mode: start workers with "
                f"'harbor worker run --job-dir {self._job_dir} "
                f"--node-name <hostname>'"
            )

        results: list[TrialResult] = []
        try:
            results = await self._monitor_loop(queue, shutdown)
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
            self._logger.info("Interrupted, shutting down workers...")
        finally:
            if not self._manual_workers:
                await self._shutdown_workers()

        return results

    async def _preflight_checks(self) -> None:
        """Verify all nodes are reachable, Docker works, and clean stale containers.

        Retries each node up to 3 times to handle transient SSH/NFS issues.
        """
        self._logger.info(f"Pre-flight: checking {len(self._nodes)} nodes...")
        max_attempts = 3
        for node in self._nodes:
            user = node.ssh_user or self._current_user
            target = f"{user}@{node.hostname}"

            check_cmd = (
                "echo OK && "
                "docker info --format '{{.ContainersRunning}}' && "
                "STALE=$(docker ps -aq --filter name=__ 2>/dev/null | wc -l) && "
                'if [ "$STALE" -gt 0 ]; then '
                "  docker ps -aq --filter name=__ | xargs docker rm -f >/dev/null 2>&1; "
                '  echo "cleaned $STALE stale containers"; '
                "else echo 'clean'; fi"
            )
            last_error = ""
            for attempt in range(max_attempts):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ssh",
                        "-o",
                        "ConnectTimeout=10",
                        target,
                        check_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=120
                    )
                    output = stdout.decode().strip().split("\n")
                    if proc.returncode != 0 or not output or output[0] != "OK":
                        last_error = stderr.decode()[:200]
                        if attempt < max_attempts - 1:
                            self._logger.warning(
                                f"  {node.hostname}: attempt {attempt + 1} "
                                f"failed, retrying..."
                            )
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        raise RuntimeError(
                            f"Node {node.hostname} pre-flight failed after "
                            f"{max_attempts} attempts: {last_error}"
                        )
                    running = output[1] if len(output) > 1 else "?"
                    cleanup = output[2] if len(output) > 2 else ""
                    self._logger.info(
                        f"  {node.hostname}: OK (docker running={running}, {cleanup})"
                    )
                    break
                except asyncio.TimeoutError:
                    last_error = "SSH timed out (120s)"
                    if attempt < max_attempts - 1:
                        self._logger.warning(
                            f"  {node.hostname}: attempt {attempt + 1} "
                            f"timed out, retrying..."
                        )
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise RuntimeError(
                            f"Node {node.hostname} pre-flight failed after "
                            f"{max_attempts} attempts: {last_error}"
                        )

    def _clean_stale_worker_state(self) -> None:
        """Remove heartbeat, log, env, and script files from previous runs.

        Without this, stale heartbeats with ``status: stopped`` cause
        the coordinator to detect 'all workers stopped' and exit early.
        """
        workers_dir = self._job_dir / ".workers"
        if not workers_dir.exists():
            return
        for f in workers_dir.iterdir():
            if f.suffix in (".json", ".log", ".env", ".sh"):
                f.unlink(missing_ok=True)
        self._logger.info("Cleaned stale worker state from previous run")

    # -- Worker launch -------------------------------------------------------

    async def _launch_workers(self) -> None:
        """SSH to each node and start a detached harbor worker process."""
        workers_dir = self._job_dir / ".workers"
        workers_dir.mkdir(parents=True, exist_ok=True)

        for node in self._nodes:
            n_conc = node.n_concurrent or self._n_concurrent_trials
            user = node.ssh_user or self._current_user

            # Collect env vars. API keys from coordinator env are passed
            # via a chmod-600 file on NFS (not on the command line).
            env_vars = dict(node.env_vars)
            for key in (
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
            ):
                val = os.environ.get(key)
                if val and key not in env_vars:
                    env_vars[key] = val

            env_file = workers_dir / f"{node.hostname}.env"
            env_lines = [f"export {k}={shlex.quote(v)}" for k, v in env_vars.items()]
            env_file.write_text("\n".join(env_lines) + "\n")
            env_file.chmod(0o600)

            log_path = workers_dir / f"{node.hostname}.log"

            # Write worker launch script to NFS to avoid shell quoting.
            script_path = workers_dir / f"{node.hostname}.sh"
            script_path.write_text(
                "#!/bin/bash\n"
                "ulimit -n 65536 2>/dev/null || true\n"
                f"source {shlex.quote(str(env_file))}\n"
                f"exec {shlex.quote(self._worker_python)} worker run "
                f"--job-dir {shlex.quote(str(self._job_dir))} "
                f"--n-concurrent {n_conc} "
                f"--node-name {shlex.quote(node.hostname)}\n"
            )
            script_path.chmod(0o755)

            remote_cmd = (
                f"setsid bash {shlex.quote(str(script_path))} "
                f"< /dev/null > {shlex.quote(str(log_path))} 2>&1 &"
            )

            ssh_cmd = [
                "ssh",
                *self._ssh_options,
                "-p",
                str(node.ssh_port),
                f"{user}@{node.hostname}",
                remote_cmd,
            ]

            self._logger.info(
                f"Launching worker on {node.hostname} (n_concurrent={n_conc})"
            )

            try:
                proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=30)
                if proc.returncode != 0:
                    self._logger.warning(
                        f"SSH to {node.hostname} failed (rc={proc.returncode})"
                    )
                else:
                    self._logger.info(f"Worker started on {node.hostname}")
            except asyncio.TimeoutError:
                self._logger.warning(f"SSH to {node.hostname} timed out (30s)")

        self._logger.info(f"Launched workers on {len(self._nodes)} nodes")

    # -- Monitoring ----------------------------------------------------------

    async def _monitor_loop(
        self,
        queue: NFSWorkQueue,
        shutdown: asyncio.Event,
    ) -> list[TrialResult]:
        """Poll NFS for results until all trials complete."""
        import time as _time

        total = len(self._trial_configs)
        seen: set[str] = set()
        monitor_start = _time.monotonic()

        progress = Progress(
            SpinnerColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task_id = progress.add_task("Running trials...", total=total)

            while len(self._trial_results) < total and not shutdown.is_set():
                await asyncio.sleep(self.POLL_INTERVAL_SEC)

                # Scan for new results.
                new_results = self._scan_results(seen)
                for result in new_results:
                    self._trial_results.append(result)
                    progress.advance(task_id)

                    if self._metrics and result.config:
                        rewards = [
                            r.verifier_result.rewards
                            if r.verifier_result is not None
                            else None
                            for r in self._trial_results
                        ]
                        source = result.config.task.source or "adhoc"
                        if source in self._metrics:
                            metric_result = self._metrics[source][0].compute(rewards)
                            name, value = next(iter(metric_result.items()))
                            progress.update(
                                task_id,
                                description=(f"{name.title()}: {value:.3f}"),
                            )

                # Check worker health.
                heartbeats = read_all_heartbeats(self._job_dir)
                expected = {n.hostname for n in self._nodes}

                for node_name, hb in heartbeats.items():
                    if node_name not in expected:
                        continue
                    if hb.status not in (
                        "stopped",
                        "draining",
                    ) and not is_worker_alive(hb, self.WORKER_TIMEOUT_SEC):
                        self._logger.warning(
                            f"Worker {node_name} appears dead, reclaiming tasks"
                        )
                        reclaimed = queue.reclaim_from_dead_worker(node_name)
                        if reclaimed:
                            self._logger.info(
                                f"Reclaimed {reclaimed} tasks from {node_name}"
                            )

                # Check if all expected workers registered and stopped.
                registered = {n for n in heartbeats if n in expected}
                all_stopped = len(registered) == len(expected) and all(
                    heartbeats[n].status == "stopped" for n in registered
                )

                if all_stopped and queue.is_empty:
                    # Final scan for stragglers.
                    await asyncio.sleep(self.POLL_INTERVAL_SEC)
                    for r in self._scan_results(seen):
                        self._trial_results.append(r)
                        progress.advance(task_id)
                    if len(self._trial_results) >= total:
                        break
                    remaining = total - len(self._trial_results)
                    self._logger.warning(
                        f"All workers stopped but {remaining} trials have no results."
                    )
                    break

                # Per-node status log.
                if heartbeats:
                    parts = []
                    for nn in sorted(heartbeats):
                        if nn not in expected:
                            continue
                        hb = heartbeats[nn]
                        parts.append(
                            f"{nn.split('.')[0]}:"
                            f"{hb.status[0]}/"
                            f"c{hb.trials_claimed}/"
                            f"d{hb.trials_completed}/"
                            f"f{hb.trials_failed}"
                        )
                    solved = sum(
                        1
                        for r in self._trial_results
                        if r.verifier_result
                        and r.verifier_result.rewards
                        and next(
                            iter(r.verifier_result.rewards.values()),
                            0,
                        )
                        > 0
                    )
                    elapsed_min = (_time.monotonic() - monitor_start) / 60
                    rate = (
                        len(self._trial_results) / elapsed_min
                        if elapsed_min > 0.5
                        else 0
                    )
                    self._logger.info(
                        f"[{len(self._trial_results)}/{total} "
                        f"solved={solved} "
                        f"rate={rate:.1f}/min "
                        f"q={queue.pending_count}] "
                        f"{' | '.join(parts)}"
                    )

        return self._trial_results

    def _scan_results(self, seen: set[str]) -> list[TrialResult]:
        """Scan the job directory for new result.json files."""
        new_results: list[TrialResult] = []
        try:
            for trial_dir in self._job_dir.iterdir():
                if not trial_dir.is_dir():
                    continue
                if trial_dir.name.startswith("."):
                    continue
                if trial_dir.name in seen:
                    continue
                result_path = trial_dir / "result.json"
                if not result_path.exists():
                    continue
                try:
                    result = TrialResult.model_validate_json(result_path.read_text())
                    # Skip preliminary results (written at trial start,
                    # no finished_at yet — trial still running).
                    if result.finished_at is None:
                        continue
                    seen.add(trial_dir.name)
                    new_results.append(result)
                except Exception:
                    continue
        except OSError:
            pass
        return new_results

    # -- Shutdown ------------------------------------------------------------

    async def _shutdown_workers(self) -> None:
        """Send SIGTERM to worker processes on remote nodes."""
        heartbeats = read_all_heartbeats(self._job_dir)
        for node in self._nodes:
            user = node.ssh_user or self._current_user
            hb = heartbeats.get(node.hostname)
            if not hb or hb.status == "stopped":
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ssh",
                    "-o",
                    "ConnectTimeout=5",
                    f"{user}@{node.hostname}",
                    f"kill {hb.pid} 2>/dev/null || true",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=10)
            except Exception:
                pass

        # Clean up env files (contain API keys).
        for node in self._nodes:
            env_file = self._job_dir / ".workers" / f"{node.hostname}.env"
            env_file.unlink(missing_ok=True)
