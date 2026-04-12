"""lucent-serverless-controller: the control plane.

Runs on your laptop. Owns the pod table, runs the scheduler and reaper
loops, exposes /resolve for clients to get a pod URL.

    lucent-controller          # starts on :9000
    curl localhost:9000/apps   # list registered apps
    curl localhost:9000/pods   # list live pods
    curl localhost:9000/resolve?app_id=echo  # get (or cold-start) a pod
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI

from . import db
from .reaper import reaper_loop
from .resolver import router
from .scheduler import scheduler_loop

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    stop = threading.Event()

    sched = threading.Thread(
        target=scheduler_loop, args=(stop,), daemon=True, name="scheduler"
    )
    reap = threading.Thread(
        target=reaper_loop, args=(stop,), daemon=True, name="reaper"
    )
    sched.start()
    reap.start()
    log.info("control plane started (scheduler + reaper running)")

    yield

    stop.set()
    sched.join(timeout=5)
    reap.join(timeout=5)
    log.info("control plane stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="lucent-serverless-controller", lifespan=lifespan)
    app.include_router(router)
    return app


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    port = 9000
    app = create_app()
    log.info("controller listening on :%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
