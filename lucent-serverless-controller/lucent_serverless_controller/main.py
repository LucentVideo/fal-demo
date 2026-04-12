"""lucent-serverless-controller: the control plane.

Deployed as a persistent CPU pod on RunPod. Owns the pod table, runs
the scheduler and reaper loops, exposes /resolve for clients.

Auth: set LUCENT_API_KEY env var. All endpoints except /health require
Authorization: Bearer <key>. If LUCENT_API_KEY is unset, auth is
disabled (local dev mode).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()

from . import db
from .code_store import CODE_DIR, code_router
from .reaper import reaper_loop
from .resolver import router
from .scheduler import scheduler_loop

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    CODE_DIR.mkdir(parents=True, exist_ok=True)
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

    # Auth: exempt /health (RunPod probe) and /dashboard (has its own login)
    PUBLIC_PATHS = {"/health", "/dashboard"}
    PUBLIC_PREFIXES = ("/apps/",)  # GET /apps/{id}/code — pods fetch without API key

    def _is_public(path: str, method: str) -> bool:
        if path in PUBLIC_PATHS:
            return True
        # Allow GET on code download (pods don't have API keys)
        if method == "GET" and any(path.startswith(p) for p in PUBLIC_PREFIXES) and path.endswith("/code"):
            return True
        return False

    @app.middleware("http")
    async def check_api_key(request: Request, call_next):
        if _is_public(request.url.path, request.method):
            return await call_next(request)
        expected = os.environ.get("LUCENT_API_KEY")
        if expected:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != expected:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid or missing API key"},
                )
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "lucent-controller"}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    app.include_router(router)
    app.include_router(code_router)
    return app


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    port = int(os.environ.get("LUCENT_PORT", "8000"))
    app = create_app()
    log.info("controller listening on :%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
