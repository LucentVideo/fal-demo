"""Job-mode entrypoint: boots a user App, then polls the controller for jobs.

Sequence:
  1. Import the user module named by LUCENT_APP_MODULE
  2. Find the unique App subclass
  3. Instantiate, run setup() once (model loading)
  4. Find the @handler-decorated method
  5. Start a health endpoint on a background thread
  6. Poll GET /jobs/next on the controller in a loop
  7. On each job: call handler(input), POST result back

The health endpoint reports {"idle": true/false, "current_job_id": "..."}
so the scheduler/reaper can manage pod lifecycle.
"""

import importlib
import inspect
import json
import logging
import os
import sys
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI

from . import App, state
from .runner import find_app_class

log = logging.getLogger(__name__)


def find_handler(instance: App):
    """Return the single @handler-decorated method on the instance."""
    candidates = []
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        if getattr(method, "_lucent_handler", False):
            candidates.append((name, method))
    if not candidates:
        raise RuntimeError(
            f"no @lucent_serverless.handler method found on {type(instance).__name__}"
        )
    if len(candidates) > 1:
        names = ", ".join(n for n, _ in candidates)
        raise RuntimeError(
            f"multiple @handler methods on {type(instance).__name__}: {names}"
        )
    return candidates[0][1]


def _build_health_api(app_id: str) -> FastAPI:
    api = FastAPI()

    @api.get("/health")
    def health():
        return {
            "status": "ready",
            "app_id": app_id,
            "version": "lucent-serverless/0.1",
            **state.job_snapshot(),
        }

    return api


def _run_health_server(app_id: str, port: int) -> None:
    api = _build_health_api(app_id)
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="warning")


def _poll_loop(
    handler_fn,
    *,
    controller_url: str,
    api_key: str,
    worker_id: str,
    app_id: str,
) -> None:
    """Continuously poll the controller for jobs and execute them."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    while True:
        try:
            with httpx.Client(timeout=10.0) as c:
                resp = c.get(
                    f"{controller_url}/jobs/next",
                    params={"worker_id": worker_id, "app_id": app_id},
                    headers=headers,
                )
        except httpx.HTTPError as e:
            log.warning("poll failed: %s", e)
            time.sleep(2)
            continue

        if resp.status_code == 204:
            time.sleep(1)
            continue

        if resp.status_code != 200:
            log.warning("unexpected poll response %d: %s", resp.status_code, resp.text)
            time.sleep(2)
            continue

        job = resp.json()
        job_id = job["job_id"]
        log.info("executing job %s", job_id)
        state.job_started(job_id)

        result_body: dict
        try:
            output = handler_fn(job["input"])
            result_body = {"output": output}
        except Exception as e:
            log.exception("job %s failed", job_id)
            result_body = {"error": str(e)}
        finally:
            state.job_finished()

        try:
            with httpx.Client(timeout=30.0) as c:
                c.post(
                    f"{controller_url}/jobs/{job_id}/result",
                    json=result_body,
                    headers=headers,
                )
        except httpx.HTTPError as e:
            log.error("failed to post result for job %s: %s", job_id, e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    module_name = os.environ.get("LUCENT_APP_MODULE", "app")
    app_id = os.environ.get("LUCENT_APP_ID", module_name)
    port = int(os.environ.get("LUCENT_PORT", "8000"))
    controller_url = os.environ.get("LUCENT_CONTROLLER_URL", "")
    api_key = os.environ.get("LUCENT_API_KEY", "")
    worker_id = os.environ.get("RUNPOD_POD_ID", f"local-{os.getpid()}")

    if not controller_url:
        raise RuntimeError(
            "LUCENT_CONTROLLER_URL must be set for job workers"
        )

    sys.path.insert(0, os.getcwd())
    log.info("importing user module %r", module_name)
    module = importlib.import_module(module_name)

    cls = find_app_class(module)
    log.info("instantiating %s", cls.__name__)
    instance = cls()
    log.info("running %s.setup()", cls.__name__)
    instance.setup()

    handler_fn = find_handler(instance)
    log.info("found handler: %s.%s", cls.__name__, handler_fn.__name__)

    health_thread = threading.Thread(
        target=_run_health_server, args=(app_id, port), daemon=True,
        name="health-server",
    )
    health_thread.start()
    log.info("health server on :%d, starting poll loop", port)

    _poll_loop(
        handler_fn,
        controller_url=controller_url,
        api_key=api_key,
        worker_id=worker_id,
        app_id=app_id,
    )


if __name__ == "__main__":
    main()
