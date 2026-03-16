"""NFS-based work queue for distributed trial execution.

Uses atomic ``os.rename()`` for task claiming, which is safe on NFSv3+
for same-filesystem moves.  No external dependencies (no Redis, no
message brokers).

Directory layout::

    {job_dir}/
      .queue/              unclaimed trial configs (one JSON per trial)
      .claimed/{node}/     claimed by a specific worker node
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from harbor.models.trial.config import TrialConfig

logger = logging.getLogger(__name__)


class NFSWorkQueue:
    """Filesystem-based work queue with atomic claim semantics."""

    QUEUE_DIR = ".queue"
    CLAIMED_DIR = ".claimed"

    def __init__(self, job_dir: Path) -> None:
        self._job_dir = job_dir
        self._queue_dir = job_dir / self.QUEUE_DIR
        self._claimed_dir = job_dir / self.CLAIMED_DIR

    # -- Writing (coordinator) -----------------------------------------------

    def write_task_groups(self, configs: list[TrialConfig]) -> int:
        """Write configs grouped by task for multi-rollout.

        When ``n_attempts > 1``, multiple configs share the same task.
        Writing them as a group ensures a single worker claims all
        rollouts for a task, enabling snapshot-based optimization.

        Each queue file contains a JSON array of configs for one task,
        named ``{task_name}.json``.

        Returns the number of task groups written.
        """
        import json
        from collections import defaultdict

        self._queue_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale claimed dirs.
        if self._claimed_dir.exists():
            for node_dir in self._claimed_dir.iterdir():
                if node_dir.is_dir():
                    for f in node_dir.iterdir():
                        f.unlink(missing_ok=True)

        # Group by task name.
        groups: dict[str, list[TrialConfig]] = defaultdict(list)
        for config in configs:
            task_name = config.task.get_task_id().get_name()
            groups[task_name].append(config)

        written = 0
        for task_name, group_configs in groups.items():
            # Skip if ALL rollouts have COMPLETED results (finished_at set).
            # Preliminary results (finished_at=None) from killed trials
            # should NOT count as done.
            all_done = all(
                self._is_result_complete(self._job_dir / c.trial_name / "result.json")
                for c in group_configs
            )
            if all_done:
                continue
            path = self._queue_dir / f"{task_name}.json"
            if path.exists():
                continue
            # Write array of configs.
            path.write_text(
                json.dumps(
                    [c.model_dump(mode="json") for c in group_configs],
                    indent=2,
                )
            )
            written += 1
        return written

    def claim_task_group(self, node_name: str) -> list[TrialConfig] | None:
        """Claim one task group (all rollouts for a task).

        Returns a list of ``TrialConfig`` for the task, or ``None``
        if the queue is empty.
        """
        import json

        node_dir = self._claimed_dir / node_name
        node_dir.mkdir(parents=True, exist_ok=True)

        try:
            entries = sorted(self._queue_dir.iterdir())
        except FileNotFoundError:
            return None

        for entry in entries:
            if not entry.name.endswith(".json"):
                continue
            dst = node_dir / entry.name
            try:
                os.rename(str(entry), str(dst))
            except (FileNotFoundError, OSError):
                continue
            try:
                raw = json.loads(dst.read_text())
                return [TrialConfig.model_validate(c) for c in raw]
            except Exception as e:
                logger.warning(f"Failed to parse task group {dst}: {e}")
                try:
                    os.rename(str(dst), str(entry))
                except OSError:
                    pass
        return None

    def write_configs(self, configs: list[TrialConfig]) -> int:
        """Serialize trial configs as individual JSON files in ``.queue/``.

        Skips tasks that already have a ``result.json`` in the job
        directory (from a previous/resumed run).  Also cleans stale
        ``.claimed/`` directories from previous runs.

        Returns the number of configs written.
        """
        self._queue_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale claimed dirs from previous runs.
        if self._claimed_dir.exists():
            for node_dir in self._claimed_dir.iterdir():
                if node_dir.is_dir():
                    for f in node_dir.iterdir():
                        f.unlink(missing_ok=True)

        written = 0
        for config in configs:
            # Skip tasks with completed results (resume support).
            # Preliminary results (finished_at=None) from killed trials
            # should be re-queued.
            result_path = self._job_dir / config.trial_name / "result.json"
            if self._is_result_complete(result_path):
                continue
            path = self._queue_dir / f"{config.trial_name}.json"
            if path.exists():
                continue
            path.write_text(config.model_dump_json(indent=2))
            written += 1
        return written

    # -- Claiming (worker) ---------------------------------------------------

    def claim_batch(self, node_name: str, batch_size: int = 5) -> list[TrialConfig]:
        """Atomically claim up to ``batch_size`` trials from the queue.

        Uses ``os.rename()`` which is atomic on NFS for same-filesystem
        moves.  If another worker renames the file first, we get
        ``FileNotFoundError`` and skip to the next file.

        Returns a list of claimed ``TrialConfig`` objects.
        """
        node_dir = self._claimed_dir / node_name
        node_dir.mkdir(parents=True, exist_ok=True)

        claimed: list[TrialConfig] = []
        try:
            entries = sorted(self._queue_dir.iterdir())
        except FileNotFoundError:
            return claimed

        for entry in entries:
            if len(claimed) >= batch_size:
                break
            if not entry.name.endswith(".json"):
                continue
            dst = node_dir / entry.name
            try:
                os.rename(str(entry), str(dst))
            except (FileNotFoundError, OSError):
                # Another worker claimed it, or NFS stale handle.
                continue
            try:
                config = TrialConfig.model_validate_json(dst.read_text())
                claimed.append(config)
            except Exception as e:
                logger.warning(f"Failed to parse claimed config {dst}: {e}")
                # Return the broken file to the queue so it's not lost.
                try:
                    os.rename(str(dst), str(entry))
                except OSError:
                    pass
        return claimed

    # -- Reclaim (coordinator, for dead workers) -----------------------------

    def reclaim_from_dead_worker(self, node_name: str) -> int:
        """Move all claimed trials from a dead worker back to ``.queue/``.

        Returns the number of trials reclaimed.
        """
        node_dir = self._claimed_dir / node_name
        if not node_dir.exists():
            return 0
        reclaimed = 0
        for entry in node_dir.iterdir():
            if not entry.name.endswith(".json"):
                continue
            # Only reclaim if no COMPLETED result exists.
            trial_name = entry.stem
            result_path = self._job_dir / trial_name / "result.json"
            if self._is_result_complete(result_path):
                entry.unlink(missing_ok=True)
                continue
            dst = self._queue_dir / entry.name
            try:
                os.rename(str(entry), str(dst))
                reclaimed += 1
            except OSError:
                pass
        return reclaimed

    # -- Status queries ------------------------------------------------------

    @staticmethod
    def _is_result_complete(result_path: Path) -> bool:
        """Check if a result.json exists AND has finished_at set.

        Preliminary results (written at trial start for crash recovery)
        have ``finished_at: null``.  Only completed results count as
        done for resume and reclaim logic.
        """
        if not result_path.exists():
            return False
        try:
            import json

            data = json.loads(result_path.read_text())
            return data.get("finished_at") is not None
        except Exception:
            return False

    @property
    def pending_count(self) -> int:
        """Number of unclaimed trials in the queue."""
        try:
            return sum(1 for f in self._queue_dir.iterdir() if f.name.endswith(".json"))
        except FileNotFoundError:
            return 0

    @property
    def claimed_counts(self) -> dict[str, int]:
        """Number of claimed trials per node."""
        counts: dict[str, int] = {}
        if not self._claimed_dir.exists():
            return counts
        for node_dir in self._claimed_dir.iterdir():
            if node_dir.is_dir():
                counts[node_dir.name] = sum(
                    1 for f in node_dir.iterdir() if f.name.endswith(".json")
                )
        return counts

    @property
    def is_empty(self) -> bool:
        """True when no unclaimed trials remain."""
        return self.pending_count == 0

    def cleanup_claimed(self, node_name: str, trial_name: str) -> None:
        """Remove a claimed file after the trial completes successfully."""
        path = self._claimed_dir / node_name / f"{trial_name}.json"
        path.unlink(missing_ok=True)

    def requeue_task_group(self, node_name: str, task_name: str) -> bool:
        """Move a claimed task group back to the queue.

        Used when setup fails on one node (e.g. missing Docker image)
        so another node can pick it up instead of permanently losing
        all rollouts for the task.

        Returns True if the group was successfully requeued.
        """
        claimed_path = self._claimed_dir / node_name / f"{task_name}.json"
        if not claimed_path.exists():
            return False
        dst = self._queue_dir / f"{task_name}.json"
        try:
            os.rename(str(claimed_path), str(dst))
            logger.info(f"Requeued task group {task_name} from {node_name}")
            return True
        except OSError:
            # Another worker may have already cleaned it up.
            claimed_path.unlink(missing_ok=True)
            return False
