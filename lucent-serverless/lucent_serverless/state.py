"""Process-wide runtime state shared by websocket handlers, /health, and watchdog.

The control plane will eventually poll /health to read these values; the
watchdog reads them locally to decide when to terminate. Kept as module
globals because there is exactly one runner per pod.
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
