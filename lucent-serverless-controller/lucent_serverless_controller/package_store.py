"""Store and serve the lucent-serverless wheel.

Upload:   PUT  /packages/lucent-serverless.whl  (body = raw .whl file)
Download: GET  /packages/lucent-serverless.whl  (no auth — pods fetch this)

One wheel on disk. GH Action builds and uploads on push.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import FileResponse

log = logging.getLogger(__name__)

PACKAGE_DIR = Path("packages")

package_router = APIRouter()


@package_router.put("/packages/{filename}")
async def upload_package(filename: str, request: Request):
    if not filename.endswith(".whl"):
        raise HTTPException(400, "only .whl files accepted")

    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PACKAGE_DIR / filename
    dest.write_bytes(body)
    log.info("stored package %s (%d bytes)", filename, len(body))
    return {"ok": True, "filename": filename, "size": len(body)}


@package_router.get("/packages/{filename}")
async def download_package(filename: str):
    path = PACKAGE_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"package {filename!r} not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename,
    )
