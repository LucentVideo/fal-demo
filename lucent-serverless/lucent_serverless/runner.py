"""Generic entrypoint that boots a user `App` inside the container.

Sequence:
  1. import the user module named by LUCENT_APP_MODULE
  2. find the unique App subclass in it
  3. instantiate, run setup() once
  4. walk methods for @realtime decorations and mount as websocket routes
  5. expose /health (includes active_connections + last_activity_at so the
     scheduler can reap idle pods — the worker itself does not self-kill)
  6. uvicorn on :LUCENT_PORT

Lifecycle note: earlier drafts had an in-pod watchdog call os._exit(0) on
idle. That doesn't work on RunPod — the container restart policy just
relaunches the runner immediately and the pod burns money in a loop. Pod
termination is the scheduler's job (Layer 3); the worker only exposes
state.
"""

import importlib
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import App, state

log = logging.getLogger(__name__)


def find_app_class(module) -> type[App]:
    """Return the unique App subclass defined in `module`. Loud failure otherwise."""
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, App) and obj is not App and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise RuntimeError(
            f"no lucent_serverless.App subclass found in module {module.__name__!r}"
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise RuntimeError(
            f"multiple App subclasses in {module.__name__!r}: {names}"
        )
    return candidates[0]


def register_realtime_routes(api: FastAPI, instance: App) -> int:
    """Mount every @realtime-decorated method on `instance` as a websocket route."""
    found = 0
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        path = getattr(method, "_lucent_realtime_path", None)
        if path is None:
            continue
        log.info("mounting realtime handler %s.%s -> %s", type(instance).__name__, name, path)

        async def endpoint(ws: WebSocket, _handler=method):
            await ws.accept()
            state.connection_opened()
            try:
                await _handler(ws)
            except WebSocketDisconnect:
                pass
            finally:
                state.connection_closed()

        api.add_api_websocket_route(path, endpoint)
        found += 1

    if found == 0:
        raise RuntimeError(
            f"no @lucent_serverless.realtime methods found on {type(instance).__name__}"
        )
    return found


def make_health_handler(app_id: str):
    def health() -> dict:
        return {
            "status": "ready",
            "app_id": app_id,
            "version": "lucent-serverless/0.1",
            **state.snapshot(),
        }

    return health


BOOT_TIMINGS_FILE = Path("/tmp/lucent_boot.json")


def _load_boot_timings() -> dict:
    """Read timing data written by entrypoint.sh."""
    if BOOT_TIMINGS_FILE.exists():
        try:
            return json.loads(BOOT_TIMINGS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def build_api(instance: App, app_id: str) -> FastAPI:
    api = FastAPI()
    register_realtime_routes(api, instance)
    api.get("/health")(make_health_handler(app_id))
    return api


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    module_name = os.environ.get("LUCENT_APP_MODULE", "app")
    app_id = os.environ.get("LUCENT_APP_ID", module_name)
    port = int(os.environ.get("LUCENT_PORT", "8000"))

    # Load entrypoint timings (boot_start, code_downloaded, deps_installed)
    timings = _load_boot_timings()

    sys.path.insert(0, os.getcwd())
    log.info("importing user module %r", module_name)
    module = importlib.import_module(module_name)

    cls = find_app_class(module)
    log.info("instantiating %s", cls.__name__)
    instance = cls()

    log.info("running %s.setup()", cls.__name__)
    timings["setup_start"] = time.time()
    instance.setup()
    timings["setup_done"] = time.time()

    api = build_api(instance, app_id)

    # Record ready timestamp and compute total
    timings["ready_at"] = time.time()
    boot_start = timings.get("boot_start", timings["ready_at"])
    timings["total_sec"] = round(timings["ready_at"] - boot_start, 2)
    log.info("boot complete in %.2fs", timings["total_sec"])

    state.set_boot_timings(timings)

    log.info("uvicorn listening on :%d", port)
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
