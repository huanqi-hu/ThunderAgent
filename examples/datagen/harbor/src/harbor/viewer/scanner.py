"""Scanner for discovering jobs and trials in a folder."""

import logging
from pathlib import Path

from pydantic import ValidationError

from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.result import TrialResult

logger = logging.getLogger(__name__)


class JobScanner:
    """Scans a folder for job and trial data."""

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir

    def list_jobs(self) -> list[str]:
        """List all job names (directory names) in the jobs folder.

        A directory is considered a job if it contains a ``config.json`` file.
        Directories without ``config.json`` (e.g. trial dirs when the viewer
        is pointed at a single job folder) are silently skipped.
        """
        if not self.jobs_dir.exists():
            return []
        return sorted(
            [
                d.name
                for d in self.jobs_dir.iterdir()
                if d.is_dir() and (d / "config.json").exists()
            ],
            reverse=True,  # Most recent first
        )

    def get_job_config(self, job_name: str) -> JobConfig | None:
        """Load job config from disk."""
        config_path = self.jobs_dir / job_name / "config.json"
        if not config_path.exists():
            return None
        try:
            return JobConfig.model_validate_json(config_path.read_text())
        except ValidationError:
            logger.warning("Failed to parse job config: %s", config_path)
            return None

    def get_job_result(self, job_name: str) -> JobResult | None:
        """Load job result from disk.

        Returns ``None`` when the file is missing or cannot be parsed
        (e.g. written by an older version with a different schema).
        """
        result_path = self.jobs_dir / job_name / "result.json"
        if not result_path.exists():
            return None
        try:
            return JobResult.model_validate_json(result_path.read_text())
        except ValidationError:
            logger.warning("Failed to parse job result: %s", result_path)
            return None

    def list_trials(self, job_name: str) -> list[str]:
        """List all trial names in a job folder."""
        job_dir = self.jobs_dir / job_name
        if not job_dir.exists():
            return []
        return sorted(
            [
                d.name
                for d in job_dir.iterdir()
                if d.is_dir() and (d / "result.json").exists()
            ]
        )

    def get_trial_result(self, job_name: str, trial_name: str) -> TrialResult | None:
        """Load trial result from disk."""
        result_path = self.jobs_dir / job_name / trial_name / "result.json"
        if not result_path.exists():
            return None
        try:
            return TrialResult.model_validate_json(result_path.read_text())
        except ValidationError:
            logger.warning("Failed to parse trial result: %s", result_path)
            return None
