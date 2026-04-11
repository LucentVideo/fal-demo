"""Generic entrypoint that boots a user `App` inside the container.

Sequence:
  1. import the user module named by LUCENT_APP_MODULE
  2. find the unique App subclass in it
  3. instantiate, run setup() once
  4. walk methods for @realtime decorations and mount as websocket routes
  5. expose /health
  6. start the idle watchdog
  7. uvicorn on :LUCENT_PORT
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import App, state, watchdog

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


def build_api(instance: App, app_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        task = asyncio.create_task(watchdog.run(instance.keep_alive))
        try:
            yield
        finally:
            task.cancel()

    api = FastAPI(lifespan=lifespan)
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

    sys.path.insert(0, os.getcwd())
    log.info("importing user module %r", module_name)
    module = importlib.import_module(module_name)

    cls = find_app_class(module)
    log.info("instantiating %s", cls.__name__)
    instance = cls()
    log.info("running %s.setup()", cls.__name__)
    instance.setup()

    api = build_api(instance, app_id)
    log.info("uvicorn listening on :%d", port)
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
