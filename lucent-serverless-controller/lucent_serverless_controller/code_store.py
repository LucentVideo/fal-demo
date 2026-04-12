"""Store and serve user app code tarballs.

Upload:   PUT  /apps/{app_id}/code   (body = raw tar.gz)
Download: GET  /apps/{app_id}/code   (returns tar.gz, no auth — pods fetch this)

Tarballs live on disk at CODE_DIR/{app_id}.tar.gz.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import FileResponse

from . import db

log = logging.getLogger(__name__)

CODE_DIR = Path("code")

code_router = APIRouter()


@code_router.put("/apps/{app_id}/code")
async def upload_code(app_id: str, request: Request):
    if not db.get_app(app_id):
        raise HTTPException(404, f"unknown app_id {app_id!r} — register it first")

    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")

    dest = CODE_DIR / f"{app_id}.tar.gz"
    dest.write_bytes(body)
    log.info("stored %d bytes of code for %s", len(body), app_id)
    return {"ok": True, "app_id": app_id, "size": len(body)}


@code_router.get("/apps/{app_id}/code")
async def download_code(app_id: str):
    path = CODE_DIR / f"{app_id}.tar.gz"
    if not path.exists():
        raise HTTPException(404, f"no code uploaded for {app_id!r}")
    return FileResponse(path, media_type="application/gzip", filename=f"{app_id}.tar.gz")
