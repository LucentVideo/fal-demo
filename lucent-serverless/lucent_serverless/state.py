"""Process-wide runtime state shared by handlers and /health.

Two modes of state tracking:
  - Realtime: active_connections (websocket sessions)
  - Job: idle flag + current_job_id (poll-loop workers)

The scheduler polls /health to read these values and decides when to
reap idle pods. Kept as module globals because there is exactly one
runner per pod.
"""

import threading
import time

_lock = threading.Lock()

# Realtime mode state
_active_connections = 0

# Job mode state
_idle = True
_current_job_id: str | None = None

# Shared
_last_activity_at = time.time()
_started_at = time.time()


# ── Realtime mode ─────────────────────────────────────────────────────

def connection_opened() -> None:
    global _active_connections, _last_activity_at
    with _lock:
        _active_connections += 1
        _last_activity_at = time.time()


def connection_closed() -> None:
    global _active_connections, _last_activity_at
    with _lock:
        _active_connections = max(0, _active_connections - 1)
        _last_activity_at = time.time()


def snapshot() -> dict:
    with _lock:
        return {
            "active_connections": _active_connections,
            "last_activity_at": int(_last_activity_at),
            "started_at": int(_started_at),
        }


# ── Job mode ──────────────────────────────────────────────────────────

def job_started(job_id: str) -> None:
    global _idle, _current_job_id, _last_activity_at
    with _lock:
        _idle = False
        _current_job_id = job_id
        _last_activity_at = time.time()


def job_finished() -> None:
    global _idle, _current_job_id, _last_activity_at
    with _lock:
        _idle = True
        _current_job_id = None
        _last_activity_at = time.time()


def job_snapshot() -> dict:
    with _lock:
        return {
            "idle": _idle,
            "current_job_id": _current_job_id,
            "last_activity_at": int(_last_activity_at),
            "started_at": int(_started_at),
        }
