"""Process-wide runtime state shared by websocket handlers and /health.

The scheduler (Layer 3) polls /health to read these values and decides
when to reap idle pods. The worker itself does not self-terminate —
RunPod's container restart policy turns in-pod self-kill into a loop.
Kept as module globals because there is exactly one runner per pod.
"""

import threading
import time

_lock = threading.Lock()
_active_connections = 0
_last_activity_at = time.time()
_started_at = time.time()


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
