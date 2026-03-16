"""Worker heartbeat protocol for distributed execution.

Workers write a JSON heartbeat file every ``INTERVAL`` seconds to
``{job_dir}/.workers/{node_name}.json``.  The coordinator reads these
to detect dead workers and reclaim their tasks.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

HEARTBEAT_DIR = ".workers"
HEARTBEAT_INTERVAL_SEC = 30
WORKER_TIMEOUT_SEC = 300  # 5 minutes without heartbeat = dead


class WorkerHeartbeat(BaseModel):
    """Heartbeat data written by each worker."""

    hostname: str
    node_name: str
    pid: int
    started_at: str
    last_active: str
    trials_claimed: int = 0
    trials_completed: int = 0
    trials_failed: int = 0
    status: str = "starting"  # starting | running | draining | stopped


class HeartbeatWriter:
    """Writes periodic heartbeats for a worker node."""

    def __init__(self, job_dir: Path, node_name: str) -> None:
        self._dir = job_dir / HEARTBEAT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{node_name}.json"
        self._node_name = node_name
        now = datetime.now(timezone.utc).isoformat()
        self._heartbeat = WorkerHeartbeat(
            hostname=platform.node(),
            node_name=node_name,
            pid=os.getpid(),
            started_at=now,
            last_active=now,
        )
        self._write()

    def update(self, **fields: object) -> None:
        """Update heartbeat fields and write to disk."""
        self._heartbeat = self._heartbeat.model_copy(
            update={
                **fields,
                "last_active": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write()

    def inc_completed(self) -> None:
        """Increment the completed counter and write."""
        self.update(trials_completed=self._heartbeat.trials_completed + 1)

    def inc_failed(self) -> None:
        """Increment the failed counter and write."""
        self.update(trials_failed=self._heartbeat.trials_failed + 1)

    def _write(self) -> None:
        try:
            self._path.write_text(self._heartbeat.model_dump_json(indent=2))
        except OSError as e:
            logger.debug(f"Heartbeat write failed: {e}")

    async def run_loop(self) -> None:
        """Background task: update ``last_active`` every interval."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
                self.update()
        except asyncio.CancelledError:
            self.update(status="stopped")


def read_all_heartbeats(job_dir: Path) -> dict[str, WorkerHeartbeat]:
    """Read all worker heartbeats from the job directory."""
    hb_dir = job_dir / HEARTBEAT_DIR
    if not hb_dir.exists():
        return {}
    result: dict[str, WorkerHeartbeat] = {}
    for path in hb_dir.iterdir():
        if not path.name.endswith(".json"):
            continue
        try:
            hb = WorkerHeartbeat.model_validate_json(path.read_text())
            result[hb.node_name] = hb
        except Exception:
            continue
    return result


def is_worker_alive(
    heartbeat: WorkerHeartbeat,
    timeout_sec: float = WORKER_TIMEOUT_SEC,
) -> bool:
    """Check if a worker is alive based on its last heartbeat."""
    try:
        last = datetime.fromisoformat(heartbeat.last_active)
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() < timeout_sec
    except Exception:
        return False
