"""Distributed worker: claims tasks from an NFS queue, runs them locally.

Launched by ``DistributedOrchestrator`` via SSH, or run manually::

    harbor worker run --job-dir /nfs/jobs/my-eval --n-concurrent 128
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from typer import Option, Typer

if TYPE_CHECKING:
    from harbor.distributed.heartbeat import HeartbeatWriter
    from harbor.distributed.queue import NFSWorkQueue
    from harbor.models.job.config import RetryConfig
    from harbor.models.trial.config import TrialConfig

worker_app = Typer(no_args_is_help=True)


@worker_app.command("run")
def run_worker(
    job_dir: Annotated[
        Path,
        Option("--job-dir", help="Path to the shared job directory on NFS"),
    ],
    n_concurrent: Annotated[
        int,
        Option("--n-concurrent", help="Max concurrent trials on this node"),
    ] = 48,
    batch_size: Annotated[
        int,
        Option("--batch-size", help="Trials to claim per batch"),
    ] = 5,
    node_name: Annotated[
        str | None,
        Option("--node-name", help="Override node name (default: hostname)"),
    ] = None,
) -> None:
    """Run as a distributed worker, claiming tasks from the NFS queue."""
    import platform

    node = node_name or platform.node().split(".")[0]

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{node}] %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    asyncio.run(_worker_main(job_dir, n_concurrent, batch_size, node))


async def _worker_main(
    job_dir: Path,
    n_concurrent: int,
    batch_size: int,
    node_name: str,
) -> None:
    from harbor.distributed.heartbeat import HeartbeatWriter
    from harbor.distributed.queue import NFSWorkQueue
    from harbor.environments.docker.docker import DockerEnvironment

    log = logging.getLogger(f"worker.{node_name}")
    t0 = time.monotonic()
    log.info(
        f"Starting: n_concurrent={n_concurrent}, "
        f"batch_size={batch_size}, job_dir={job_dir}"
    )

    queue = NFSWorkQueue(job_dir)
    heartbeat = HeartbeatWriter(job_dir, node_name)
    heartbeat.update(status="starting")

    retry_config = _load_retry_config(job_dir)

    await _prebuild_agents(job_dir, log)
    await _cleanup_stale_containers(log)

    DockerEnvironment.configure(
        max_concurrent_lifecycle=min(n_concurrent, 64),
        max_concurrent_builds=min(n_concurrent, 32),
        stagger_ms=max(100, min(n_concurrent * 3, 500)),
    )
    await DockerEnvironment.start_garbage_collector()

    semaphore = asyncio.Semaphore(n_concurrent)
    active_tasks: set[asyncio.Task] = set()
    completed = 0
    failed = 0
    claimed_total = 0
    draining = False
    # Multi-rollout backpressure: 0 = not yet determined, >0 = limit.
    max_in_flight_groups = 0
    # Track tasks that failed setup on this node to avoid requeue loops.
    setup_failed_tasks: set[str] = set()

    heartbeat.update(status="running")
    log.info(f"Ready, queue has {queue.pending_count} pending tasks")

    def handle_sigterm(*_: object) -> None:
        nonlocal draining
        log.info("SIGTERM received, draining...")
        draining = True

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

    hb_task = asyncio.create_task(heartbeat.run_loop())

    try:
        while not draining:
            # Reap finished tasks.
            done = {t for t in active_tasks if t.done()}
            for t in done:
                if t.exception():
                    log.error(f"Task error: {t.exception()}")
                    failed += 1
                else:
                    completed += 1
            active_tasks -= done

            # ── Multi-rollout: backpressure + claim ──
            # When at capacity, wait for a group to finish before claiming.
            if max_in_flight_groups > 0 and len(active_tasks) >= max_in_flight_groups:
                if active_tasks:
                    done_wait, _ = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done_wait:
                        if t.exception():
                            log.error(f"Task error: {t.exception()}")
                            failed += 1
                        else:
                            completed += 1
                    active_tasks -= done_wait
                continue

            task_group = queue.claim_task_group(node_name)
            if task_group and len(task_group) > 1:
                # Skip tasks that already failed setup on this node
                # to avoid requeue loops (claim → fail → requeue → claim).
                group_task = task_group[0].task.get_task_id().get_name()
                if group_task in setup_failed_tasks:
                    queue.requeue_task_group(node_name, group_task)
                    await asyncio.sleep(0.5)
                    continue

                # Set max_in_flight on first multi-rollout group.
                if max_in_flight_groups == 0:
                    n_attempts = len(task_group)
                    # Groups to saturate LLM + pipeline depth for setup.
                    groups_for_saturation = (
                        n_concurrent + n_attempts - 1
                    ) // n_attempts
                    max_in_flight_groups = groups_for_saturation
                    log.info(
                        f"Multi-rollout: {n_attempts} rollouts/group, "
                        f"max {max_in_flight_groups} groups in-flight"
                    )

                claimed_total += len(task_group)
                heartbeat.update(trials_claimed=claimed_total)
                task = asyncio.create_task(
                    _execute_multi_rollout(
                        task_group,
                        semaphore,
                        retry_config,
                        queue,
                        node_name,
                        log,
                        heartbeat,
                        setup_failed_tasks,
                    )
                )
                active_tasks.add(task)
                log.info(
                    f"[{time.monotonic() - t0:.0f}s] "
                    f"claimed group ({len(task_group)} rollouts) "
                    f"active={len(active_tasks)} "
                    f"q={queue.pending_count}"
                )
                # Slow initial claims for cross-node balance.
                sleep = 1.0 if len(active_tasks) <= 4 else 0.1
                await asyncio.sleep(sleep)
                continue

            # In multi-rollout mode but no groups available: wait or exit.
            if max_in_flight_groups > 0:
                if active_tasks:
                    done_wait, _ = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done_wait:
                        if t.exception():
                            log.error(f"Task error: {t.exception()}")
                            failed += 1
                        else:
                            completed += 1
                    active_tasks -= done_wait
                elif queue.is_empty:
                    break
                else:
                    await asyncio.sleep(0.5)
                continue

            # ── Single trial: claim individual configs ──
            if task_group:
                # Single-config task group — treat as batch of 1.
                batch = task_group
            else:
                in_flight = len(active_tasks)
                free_slots = n_concurrent - in_flight
                if free_slots <= 0:
                    if active_tasks:
                        done, active_tasks = await asyncio.wait(
                            active_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in done:
                            if t.exception():
                                failed += 1
                            else:
                                completed += 1
                    continue

                effective_batch = 1 if claimed_total < n_concurrent else batch_size
                claim_n = min(effective_batch, free_slots)
                batch = queue.claim_batch(node_name, claim_n)

            if not batch and not active_tasks:
                if queue.is_empty:
                    break
                await asyncio.sleep(0.5)
                continue

            if not batch:
                if active_tasks:
                    done, active_tasks = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in done:
                        if t.exception():
                            failed += 1
                        else:
                            completed += 1
                continue

            claimed_total += len(batch)
            heartbeat.update(trials_claimed=claimed_total)

            if claimed_total <= n_concurrent:
                await asyncio.sleep(0.2)

            for config in batch:
                task = asyncio.create_task(
                    _execute_trial(
                        config,
                        semaphore,
                        retry_config,
                        queue,
                        node_name,
                        log,
                        heartbeat,
                    )
                )
                active_tasks.add(task)

            log.info(
                f"[{time.monotonic() - t0:.0f}s] "
                f"claimed={claimed_total} active={len(active_tasks)} "
                f"done={completed} failed={failed} "
                f"q={queue.pending_count}"
            )

    except KeyboardInterrupt:
        log.info("Interrupted, draining...")
        draining = True

    # Drain remaining.
    if active_tasks:
        heartbeat.update(status="draining")
        log.info(f"Draining {len(active_tasks)} active tasks...")
        done, _ = await asyncio.wait(active_tasks)
        for t in done:
            if t.exception():
                failed += 1
            else:
                completed += 1

    hb_task.cancel()
    try:
        await hb_task
    except asyncio.CancelledError:
        pass
    await DockerEnvironment.stop_garbage_collector()

    elapsed = time.monotonic() - t0
    heartbeat.update(
        status="stopped",
        trials_completed=completed,
        trials_failed=failed,
    )
    log.info(
        f"Finished in {elapsed:.0f}s: claimed={claimed_total}, "
        f"completed={completed}, failed={failed}"
    )


# ── Multi-rollout execution ──────────────────────────────────────


async def _execute_multi_rollout(
    group_configs: list[TrialConfig],
    semaphore: asyncio.Semaphore,
    retry_config: RetryConfig,
    queue: NFSWorkQueue,
    node_name: str,
    log: logging.Logger,
    heartbeat: HeartbeatWriter | None = None,
    setup_failed_tasks: set[str] | None = None,
) -> None:
    """Run all rollouts for one task using snapshot-based multi-rollout.

    Setup runs outside the LLM semaphore (I/O-bound).
    Each rollout acquires the semaphore individually (LLM-bound).
    Docker's internal semaphores handle daemon concurrency.
    """
    from harbor.trial.multi_rollout import MultiRolloutTrial

    task_name = group_configs[0].task.get_task_id().get_name()
    mrt = MultiRolloutTrial(group_configs)
    t_group_start = time.monotonic()

    setup_succeeded = False
    try:
        log.info(f"[{task_name}] Setting up + snapshotting...")
        t_setup_start = time.monotonic()
        await mrt.setup_and_snapshot()
        setup_sec = time.monotonic() - t_setup_start
        setup_succeeded = True
        log.info(f"[{task_name}] Setup + snapshot: {setup_sec:.1f}s")

        n_solved = 0
        n_failed = 0

        def on_result(result):
            nonlocal n_solved, n_failed
            reward = None
            if result.verifier_result and result.verifier_result.rewards:
                reward = next(iter(result.verifier_result.rewards.values()), None)
            if result.exception_info:
                n_failed += 1
                log.warning(
                    f"[{result.trial_name}] {result.exception_info.exception_type}"
                )
            elif reward and reward > 0:
                n_solved += 1
                log.info(f"[{result.trial_name}] SOLVED")
            else:
                log.info(f"[{result.trial_name}] reward={reward}")
            if heartbeat:
                if result.exception_info:
                    heartbeat.inc_failed()
                else:
                    heartbeat.inc_completed()

        t_rollout_start = time.monotonic()
        await mrt.run_rollouts(semaphore, on_result)
        rollout_sec = time.monotonic() - t_rollout_start
        total_sec = time.monotonic() - t_group_start
        log.info(
            f"[{task_name}] Group done: "
            f"setup={setup_sec:.0f}s rollouts={rollout_sec:.0f}s "
            f"total={total_sec:.0f}s "
            f"solved={n_solved}/{len(group_configs)} "
            f"errors={n_failed}"
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if not setup_succeeded:
            # Setup failed (e.g. missing Docker image, build error).
            # Requeue the group so another node can try it, and track
            # the failure to avoid re-claiming the same group (loop).
            log.warning(
                f"[{task_name}] Setup failed, requeueing: {type(e).__name__}: {e}"
            )
            if setup_failed_tasks is not None:
                setup_failed_tasks.add(task_name)
            queue.requeue_task_group(node_name, task_name)
            if heartbeat:
                heartbeat.inc_failed()
        else:
            log.error(f"[{task_name}] Multi-rollout failed: {e}")
            # Write failed results for rollouts that didn't run,
            # so the coordinator doesn't wait for missing results.
            from harbor.models.trial.result import (
                AgentInfo,
                ExceptionInfo,
                TrialResult,
            )

            for config in group_configs:
                result_path = config.trials_dir / config.trial_name / "result.json"
                if not result_path.exists():
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    failed_result = TrialResult(
                        trial_name=config.trial_name,
                        task_name=task_name,
                        task_id=config.task.get_task_id(),
                        trial_uri="",
                        task_checksum="",
                        config=config,
                        agent_info=AgentInfo(
                            name=config.agent.name or "unknown",
                            version="unknown",
                        ),
                        exception_info=ExceptionInfo.from_exception(e),
                    )
                    result_path.write_text(failed_result.model_dump_json(indent=2))
                    if heartbeat:
                        heartbeat.inc_failed()
    finally:
        await mrt.cleanup()
        if setup_succeeded:
            queue.cleanup_claimed(node_name, task_name)


# ── Single trial execution ───────────────────────────────────────


async def _execute_trial(
    config: TrialConfig,
    semaphore: asyncio.Semaphore,
    retry_config: RetryConfig,
    queue: NFSWorkQueue,
    node_name: str,
    log: logging.Logger,
    heartbeat: HeartbeatWriter | None = None,
) -> None:
    """Execute a single trial with retries.

    Semaphore held during setup + agent, released for finalize.
    """
    from harbor.models.trial.result import ExceptionInfo
    from harbor.trial.trial import Trial

    result_path = config.trials_dir / config.trial_name / "result.json"
    if result_path.exists():
        log.info(f"[{config.trial_name}] Already has result, skipping")
        queue.cleanup_claimed(node_name, config.trial_name)
        if heartbeat:
            heartbeat.inc_completed()
        return

    for attempt in range(retry_config.max_retries + 1):
        trial_dir = config.trials_dir / config.trial_name
        if trial_dir.exists():
            _cleanup_trial_dir(trial_dir)

        trial = Trial(config)

        try:
            async with semaphore:
                log.info(f"[{config.trial_name}] Setting up...")
                await trial.setup()
                log.info(f"[{config.trial_name}] Running agent...")
                await trial.run_agent()

            log.info(f"[{config.trial_name}] Verifying...")
            await trial.finalize()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if trial.result.exception_info is None:
                trial.result.exception_info = ExceptionInfo.from_exception(e)
            try:
                await trial._cleanup_and_finalize()
            except Exception:
                pass

        result = trial.result

        reward = None
        if result.verifier_result and result.verifier_result.rewards:
            reward = next(iter(result.verifier_result.rewards.values()), None)
        if result.exception_info:
            log.warning(f"[{config.trial_name}] {result.exception_info.exception_type}")
        elif reward and reward > 0:
            log.info(f"[{config.trial_name}] SOLVED")
        else:
            log.info(f"[{config.trial_name}] reward={reward}")

        if result.exception_info is None:
            queue.cleanup_claimed(node_name, config.trial_name)
            if heartbeat:
                heartbeat.inc_completed()
            return

        if not _should_retry(result.exception_info.exception_type, retry_config):
            queue.cleanup_claimed(node_name, config.trial_name)
            if heartbeat:
                heartbeat.inc_failed()
            return

        if attempt == retry_config.max_retries:
            queue.cleanup_claimed(node_name, config.trial_name)
            if heartbeat:
                heartbeat.inc_failed()
            return

        log.info(
            f"[{config.trial_name}] Retrying ({result.exception_info.exception_type})"
        )
        _cleanup_trial_dir(trial.trial_dir)
        delay = retry_config.min_wait_sec * (retry_config.wait_multiplier**attempt)
        delay = min(delay, retry_config.max_wait_sec)
        await asyncio.sleep(delay)


# ── Helpers ──────────────────────────────────────────────────────


def _should_retry(exception_type: str, retry_config: RetryConfig) -> bool:
    if (
        retry_config.exclude_exceptions
        and exception_type in retry_config.exclude_exceptions
    ):
        return False
    if (
        retry_config.include_exceptions
        and exception_type not in retry_config.include_exceptions
    ):
        return False
    return True


def _cleanup_trial_dir(trial_dir: Path) -> None:
    try:
        shutil.rmtree(trial_dir)
    except (PermissionError, OSError):
        try:
            subprocess.run(
                ["sudo", "rm", "-rf", str(trial_dir)],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass


def _load_retry_config(job_dir: Path) -> RetryConfig:
    from harbor.models.job.config import RetryConfig

    config_path = job_dir / "config.json"
    if config_path.exists():
        import json

        try:
            job_cfg = json.loads(config_path.read_text())
            retry_cfg = job_cfg.get("orchestrator", {}).get("retry") or {}
            if retry_cfg:
                return RetryConfig(**retry_cfg)
        except Exception:
            pass
    return RetryConfig()


async def _cleanup_stale_containers(log: logging.Logger) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-aq",
        "--filter",
        "name=__",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    ids = stdout.decode().strip().split()
    if ids and ids[0]:
        log.info(f"Cleaning {len(ids)} stale containers from previous runs")
        await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            *ids,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(2)


async def _prebuild_agents(job_dir: Path, log: logging.Logger) -> None:
    from jinja2 import Environment as JinjaEnv

    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.trial.config import TrialConfig

    config: TrialConfig | None = None
    for search_dir in [job_dir / ".queue", job_dir / ".claimed"]:
        try:
            for item in search_dir.rglob("*.json"):
                raw = json.loads(item.read_text())
                # Multi-rollout groups are stored as JSON arrays.
                if isinstance(raw, list) and raw:
                    raw = raw[0]
                config = TrialConfig.model_validate(raw)
                break
        except Exception:
            pass
        if config:
            break
    if config is None:
        return

    agent = AgentFactory.create_agent_from_config(
        config.agent, logs_dir=Path("/tmp/prebuild-probe")
    )
    if not isinstance(agent, BaseInstalledAgent):
        return

    prebuild_vols = agent._prebuild_volumes
    if not prebuild_vols:
        return

    marker = agent._cache_marker_path
    if marker:
        container_dir = str(Path(marker).parent)
        marker_name = Path(marker).name
        for host_path, container_path in prebuild_vols.items():
            if container_path == container_dir:
                if (Path(host_path) / marker_name).exists():
                    log.info("Agent cache already built, skipping prebuild")
                    return

    log.info("Pre-building agent environment...")

    template_path = agent._install_agent_template_path
    env = JinjaEnv()
    template = env.from_string(template_path.read_text())
    script = template.render(**agent._template_variables)

    vol_args: list[str] = []
    for host_path, container_path in prebuild_vols.items():
        vol_args.extend(["-v", f"{host_path}:{container_path}:rw"])
    for host_path, container_spec in agent._extra_volumes.items():
        vol_args.extend(["-v", f"{host_path}:{container_spec}"])

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        *vol_args,
        agent._prebuild_docker_image,
        "bash",
        "-c",
        script,
    ]

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)

    if proc.returncode != 0:
        log.warning(
            f"Prebuild failed (rc={proc.returncode}), "
            f"trials will install individually.\n"
            f"{stdout.decode()[-500:]}"
        )
    else:
        log.info("Agent prebuild complete")
