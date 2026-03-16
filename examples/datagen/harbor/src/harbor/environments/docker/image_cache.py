"""LRU image cache manager for Docker images.

Ensures the total disk usage of Docker images stays within a configurable
budget by evicting least-recently-used images when space is needed.  Images
with active references (running containers) are protected from eviction.

Usage::

    cache = ImageCacheManager(budget_gb=500)
    await cache.ensure("swebench/sweb.eval.x86_64.django:latest")
    await cache.acquire("swebench/sweb.eval.x86_64.django:latest")
    # ... run containers ...
    await cache.release("swebench/sweb.eval.x86_64.django:latest")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


class ImageCacheManager:
    """LRU cache for Docker images with disk budget enforcement.

    Args:
        budget_gb: Maximum disk space (in GB) to use for Docker images.
            When exceeded, LRU images with zero active refs are evicted.
        pull_semaphore: Optional semaphore to rate-limit concurrent pulls.
    """

    def __init__(
        self,
        budget_gb: float = 500.0,
        pull_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._budget_bytes = int(budget_gb * 1024**3)
        self._pull_semaphore = pull_semaphore
        self._lock = asyncio.Lock()
        self._active_refs: dict[str, int] = {}
        self._last_used: dict[str, float] = {}

    async def ensure(self, image: str) -> None:
        """Ensure image is locally available.  Pull if needed, evict LRU if over budget."""
        if await self._image_exists(image):
            self._last_used[image] = time.monotonic()
            return

        # Evict before pulling to make room.
        await self._evict_until_budget()

        # Pull (rate-limited if semaphore provided).
        if self._pull_semaphore:
            async with self._pull_semaphore:
                if not await self._image_exists(image):
                    await self._pull_image(image)
        else:
            await self._pull_image(image)

        self._last_used[image] = time.monotonic()

    async def acquire(self, image: str) -> None:
        """Protect image from eviction (increment refcount)."""
        async with self._lock:
            self._active_refs[image] = self._active_refs.get(image, 0) + 1
            self._last_used[image] = time.monotonic()

    async def release(self, image: str) -> None:
        """Allow eviction (decrement refcount)."""
        async with self._lock:
            current = self._active_refs.get(image, 1)
            if current <= 1:
                self._active_refs.pop(image, None)
            else:
                self._active_refs[image] = current - 1

    async def _evict_until_budget(self) -> None:
        """Evict LRU images until disk usage is under budget."""
        async with self._lock:
            usage = await self._get_total_usage()
            while usage > self._budget_bytes:
                victim = self._find_lru_evictable()
                if victim is None:
                    logger.warning(
                        "Image cache over budget (%d GB / %d GB) but no "
                        "evictable images (all have active refs).",
                        usage // (1024**3),
                        self._budget_bytes // (1024**3),
                    )
                    break
                logger.info("Evicting image %s (LRU)", victim)
                await self._remove_image(victim)
                self._last_used.pop(victim, None)
                usage = await self._get_total_usage()

    def _find_lru_evictable(self) -> str | None:
        """Find least-recently-used image with zero active refs."""
        candidates = [
            (img, ts)
            for img, ts in self._last_used.items()
            if self._active_refs.get(img, 0) == 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    @staticmethod
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

    @staticmethod
    async def _pull_image(image: str) -> None:
        logger.info("Pulling image %s", image)
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "pull",
            image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to pull {image}: {stderr.decode()[:300]}")

    @staticmethod
    async def _remove_image(image: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rmi",
            "-f",
            image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    @staticmethod
    async def _get_total_usage() -> int:
        """Get total Docker image disk usage in bytes."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "system",
            "df",
            "--format",
            "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return 0
        try:
            # docker system df --format json outputs one JSON object per line
            for line in stdout.decode().strip().splitlines():
                data = json.loads(line)
                if data.get("Type") == "Images":
                    # Size is human-readable like "15.2GB" — parse it
                    size_str = data.get("Size", "0B")
                    return _parse_docker_size(size_str)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return 0


def _parse_docker_size(size_str: str) -> int:
    """Parse Docker's human-readable size strings (e.g., '15.2GB') to bytes."""
    size_str = size_str.strip().upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            num = float(size_str[: -len(suffix)])
            return int(num * mult)
    return 0
