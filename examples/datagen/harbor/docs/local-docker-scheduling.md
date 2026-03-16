# Local Docker Scheduling: Architecture & Optimizations

## Overview

Harbor's local orchestrator runs agent trials in Docker containers. The
scheduling challenge: maximize LLM server utilization while managing
hundreds of concurrent containers across image pull/build, container
start, agent install, LLM execution, and verification phases.

## Current Architecture

### Bounded-Queue Pipeline

```
trial_configs
    │
    ▼
[Image Workers (16)] ──► prepared_queue (unbounded) ──► [Setup Workers (48)]
    │                                                        │
    │ docker pull / build                                    │ docker compose up
    │ (outside trial pipeline)                               │ agent install
    │                                                        │
    │                                                        ▼
    │                                              running_queue (bounded, 32)
    │                                                        │
    │                                                        ▼
    │                                              [Agent Workers (128)]
    │                                                        │
    │                                                        │ trial.run_agent()
    │                                                        │ LLM-bound, 2-25 min
    │                                                        │
    │                                                        ▼
    │                                              asyncio.create_task()
    │                                                        │
    │                                                        ▼
    │                                              [Finalize Tasks (sem=32)]
    │                                                        │
    │                                                        │ verification + cleanup
    │                                                        │ bounded by semaphore
    │                                                        ▼
    │                                                     results
```

### Why This Design

**Problem:** All Docker operations were inside a single trial semaphore.
Image builds (30-600s) and verification (12-61s each, downloads 76
packages) consumed LLM-bound agent slots.

**Solution:** Four independent stages with different concurrency limits:

| Stage | Workers | Bound By | Duration | Resource |
|---|---|---|---|---|
| Image pull/build | 16 | `_image_pull_semaphore` / `_build_semaphore` | 0-600s | Network/CPU |
| Container setup | 48 | Worker pool size | 10-20s | Docker daemon |
| Agent execution | 128 | Worker pool size | 100-1500s | **LLM server** |
| Verification | 32 | `_finalize_semaphore` | 12-61s | CPU/disk |

**Key insight:** Agent workers never wait on image I/O, container startup,
or verification. They only consume from `running_queue` and execute
against the LLM.

### Concurrency Primitives

**prepared_queue (unbounded):** Configs whose images are ready. Unbounded
to avoid deadlock: agent/setup workers re-enqueue retries here, and if
bounded, all workers could block on `put()` simultaneously with nobody
consuming. Safe because items are config tuples, not containers.

**running_queue (bounded, n_agents//4):** Containers that completed setup,
ready for agent work. This bounds how many idle containers exist between
setup and agent. Total containers ≈ running_queue size + active agents +
finalize tasks.

**remaining (Semaphore(0)):** Countdown latch for clean shutdown. Each
trial releases once on final completion (success or max retries exhausted).
Main loop acquires n_trials times before propagating shutdown sentinels.

**_finalize_semaphore (32):** Bounds concurrent verification. Without this,
88+ containers downloading grading packages simultaneously saturates
disk/network and stalls the entire pipeline (observed in v2/v3 runs).

### Fire-and-Forget Finalize

Agent workers do NOT await finalize. After `run_agent()`, they spawn
finalize as a background task (`asyncio.create_task`) and immediately
go back to consuming from `running_queue`. This prevents the 12-61s
verification phase from blocking LLM-bound agent slots.

The finalize task:
1. Acquires `_finalize_semaphore` (blocks if 32 already running)
2. Runs `trial.finalize()` (verification + cleanup)
3. Handles retry logic (re-enqueues to `prepared_queue` if retryable)
4. Releases `remaining` semaphore
5. Calls `on_result` for progress display

## Optimizations

### 1. Pre-build Images Outside Pipeline

`_ensure_image_ready()` runs before a trial enters the queue pipeline:

- **Prebuilt images** (`docker_image` in task.toml): `docker pull` under
  `_image_pull_semaphore` (16 concurrent). Double-check-locking to avoid
  duplicate pulls.

- **Dockerfile tasks**: Parse `FROM` line, pre-pull base image, build
  overlay (`docker build -t hb__<name>`) under `_build_semaphore` (32)
  with per-image locks. The overlay is typically trivial (uv + mkdir,
  5-10s with cached base).

### 2. PATH Fix for Non-Login Shells

Upstream changed `bash -lc` to `bash -c` for exec commands. This broke
PATH inheritance — tools installed to `~/.local/bin` (uv, pipx) are not
found. Fixed by prepending `export PATH="/root/.local/bin:/usr/local/bin:$PATH"`
to every exec command in `DockerEnvironment.exec()`.

### 3. Shared uv Download Cache

Verification scripts use `uv run parser.py` which downloads the SWE-bench
grading harness (76 packages, ~50MB). Without caching, every container
downloads them independently. Added shared host bind mount in
docker-compose-base.yaml:

```yaml
- ${HOST_UV_CACHE_PATH}:/root/.cache/uv
```

First container downloads, all subsequent hit the cache.

### 4. SWE-bench False Negative Fix

The `parse_log_pytest_v2` parser can't detect test results from tox-wrapped
pytest output that uses dot notation. Fixed by setting
`PYTEST_ADDOPTS=-rA` before test commands, forcing verbose summary output.

### 5. Infrastructure Error Detection

Verification exit codes 127 (command not found) and 126 (permission denied)
write `INFRA_ERROR` to reward.txt instead of `0`. Harbor's verifier raises
`VerifierOutputParseError`, surfacing infrastructure failures as explicit
errors instead of silent false-negative rewards.

### 6. Agent Install Always Runs

Removed the cache marker early-return in `BaseInstalledAgent.setup()`.
Agent-level system deps (tmux for OpenHands, Node.js for Claude Code)
can't be cached in shared volumes — they must be checked/installed per
container. The install script's own fast path handles cache checking.

### 7. environment/init.sh Convention

File-based post-start initialization. After `docker compose up`, if
`environment/init.sh` exists in the task directory, it's uploaded and
executed. Used by SWESmith for `git checkout`. Convention-based, no
schema changes needed.

## Measured Performance

### 500-task SWE-bench Verified, 128 workers (v4 run)

| Metric | Value |
|---|---|
| Solve rate | 43.3% (69/159 at time of measurement, still running) |
| R2E-Gym reference | 43.1% |
| LLM request rate | 13-22 req/s sustained |
| Agents in-flight | 128 (full saturation) |
| Infrastructure errors | 0 |
| Pipeline stalls | 0 |
| Setup time (cached) | 10-20s per trial |
| Agent time | 100-1500s per trial (median ~260s) |
| Verification time | 12-61s per trial |

### Previous runs for comparison

| Run | Design | Result |
|---|---|---|
| v2 (single semaphore) | Finalize blocks agent workers | Stalled at 42/500 (88 concurrent verifications) |
| v3 (split semaphore) | Unbounded container stampede | 208 containers, 70+ idle |
| v4 (bounded queues) | Fire-and-forget finalize | 128 agents sustained, no stall |

### Pipeline timing breakdown (from 50-task run)

```
image_ready:  min=0.0  p50=0.0  max=0.0   (all cached)
setup:        min=9.9  p50=17.7 max=20.7
agent:        min=96.8 p50=258  max=1463
total:        min=115  p50=275  max=1480
```

### LLM efficiency projections

| Workload | Workers | Tasks | Efficiency |
|---|---|---|---|
| Smoke test (50=50) | 50 | 50 | 22% (no backlog) |
| Eval (500 tasks) | 128 | 500 | ~75% (long tail) |
| Datagen (25k tasks) | 128 | 25,000 | ~92% (taper only last 0.5%) |

## Configuration

All concurrency limits scale with `n_concurrent_trials`:

```python
n_image = min(n_concurrent_trials, 16)
n_setup = min(n_concurrent_trials, 48)
n_agents = n_concurrent_trials  # e.g., 128
running_queue_size = min(n_agents // 4, 32)
finalize_semaphore = min(n_concurrent_trials, 32)
build_semaphore = min(n_concurrent_trials, 32)
lifecycle_semaphore = min(n_concurrent_trials, 64)
pull_semaphore = min(n_concurrent_trials, 16)
```

## Future Work

- **Bounded queues replacing all semaphores:** Replace Docker-level
  semaphores (build, lifecycle, pull) with queue stages.

- **Compiled agent binaries:** PEX/bun-build for agents, mounted into
  containers. Eliminates agent install phase entirely.

- **Image deduplication:** `uv` + `RUN --mount=type=cache` for shared
  package layers across images. Critical for 25k-image datagen.

- **ImageCacheManager integration:** LRU eviction with refcounting
  (module exists at `image_cache.py`, not yet wired into pipeline).

- **Host-side verification:** Run SWE-bench grading parser on the host
  instead of inside containers. Eliminates in-container package downloads.
