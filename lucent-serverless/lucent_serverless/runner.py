"""Generic entrypoint that boots a user `App` inside the container.

Sequence:
  1. import the user module named by LUCENT_APP_MODULE
  2. find the unique App subclass in it
  3. instantiate, run setup() once
  4. walk methods for @realtime decorations and mount as websocket routes
  5. expose /health (includes active_connections + last_activity_at so the
     scheduler can reap idle pods — the worker itself does not self-kill)
  6. uvicorn on :LUCENT_PORT

Realtime handlers use the fal-compatible AsyncIterator pattern:

    @realtime("/path")
    async def handler(self, inputs: AsyncIterator[Input]) -> AsyncIterator[Output]:
        async for msg in inputs:
            yield SomeOutput(...)

The runner wraps the raw websocket into async iterators automatically:
- inputs: each ws.receive_json() is parsed into the Input pydantic model
- yields: each yielded Output is serialized via .model_dump() and sent as JSON
"""

import importlib
import inspect
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import msgpack
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


def _get_input_model(method):
    """Extract the pydantic model type from the inputs: AsyncIterator[Model] hint."""
    try:
        hints = get_type_hints(method)
    except Exception:
        return None

    for param_name, hint in hints.items():
        if param_name in ("self", "return"):
            continue
        origin = get_origin(hint)
        if origin is AsyncIterator:
            args = get_args(hint)
            if args:
                return args[0]
    return None


async def _ws_input_iterator(ws: WebSocket, input_model, fmt: dict):
    """Wrap a WebSocket into an async iterator that yields parsed pydantic models.

    Accepts either text JSON frames or binary msgpack frames. Records the
    format of the first frame on `fmt["kind"]` so the sender can mirror it.
    """
    while True:
        try:
            message = await ws.receive()
        except WebSocketDisconnect:
            return
        if message.get("type") == "websocket.disconnect":
            return

        if message.get("bytes") is not None:
            raw = msgpack.unpackb(message["bytes"], raw=False)
            fmt.setdefault("kind", "msgpack")
        elif message.get("text") is not None:
            raw = json.loads(message["text"])
            fmt.setdefault("kind", "json")
        else:
            continue

        if input_model is not None:
            try:
                parsed = input_model.model_validate(raw)
            except Exception:
                # If it's a RootModel, try validating as the root type
                try:
                    parsed = input_model(root=raw) if hasattr(input_model, 'root') else input_model(**raw)
                except Exception:
                    parsed = raw
            yield parsed
        else:
            yield raw


def register_realtime_routes(api: FastAPI, instance: App) -> int:
    """Mount every @realtime-decorated method on `instance` as a websocket route."""
    found = 0
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        path = getattr(method, "_lucent_realtime_path", None)
        if path is None:
            continue
        log.info("mounting realtime handler %s.%s -> %s", type(instance).__name__, name, path)

        input_model = _get_input_model(method)

        def _make_endpoint(_handler, _input_model):
            # Close over handler/input_model instead of using default args:
            # FastAPI treats default-valued endpoint params as query params
            # and deepcopies their defaults on every request. A bound method
            # default drags the whole App instance (locks, models) through
            # deepcopy, which blows up with "cannot pickle _thread.lock".
            async def endpoint(ws: WebSocket):
                await ws.accept()
                state.connection_opened()
                fmt: dict = {}
                try:
                    inputs = _ws_input_iterator(ws, _input_model, fmt)
                    async for output in _handler(inputs):
                        if output is None:
                            continue
                        if hasattr(output, "model_dump"):
                            payload = output.model_dump()
                        elif isinstance(output, dict):
                            payload = output
                        else:
                            await ws.send_text(str(output))
                            continue
                        # Mirror the client's frame format. Default to msgpack
                        # if the client hasn't sent anything yet.
                        if fmt.get("kind", "msgpack") == "msgpack":
                            await ws.send_bytes(msgpack.packb(payload, use_bin_type=True))
                        else:
                            await ws.send_json(payload)
                except WebSocketDisconnect:
                    pass
                finally:
                    state.connection_closed()
            return endpoint

        api.add_api_websocket_route(path, _make_endpoint(method, input_model))
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
