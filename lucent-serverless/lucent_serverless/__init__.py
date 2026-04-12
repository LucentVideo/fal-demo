"""lucent-serverless: a small serverless realtime platform.

The user-facing surface:
  - App: base class — subclass it, set attributes, override setup()
  - @realtime(path): decorator to mark websocket handlers
  - deploy(AppClass): register your app with the controller
  - resolve(app_id): get a pod URL (cold-starts if needed)
  - connect(app_id, path): resolve + open websocket in one call
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

# The controller runs as a persistent CPU pod on RunPod.
# Override via LUCENT_CONTROLLER_URL env var.
CONTROLLER_URL: str = os.environ.get(
    "LUCENT_CONTROLLER_URL",
    "https://dxdfhdmmgl2wtf-8000.proxy.runpod.net",
)


class SpawnInfo:
    """Returned by App.spawn() — holds the app_id and pod_url."""

    def __init__(self, app_id: str, pod_url: str) -> None:
        self.app_id = app_id
        self.pod_url = pod_url

    def __repr__(self) -> str:
        return f"SpawnInfo(app_id={self.app_id!r}, pod_url={self.pod_url!r})"


class App:
    """Base class for a lucent-serverless app.

    Subclass this, set the class attributes you care about, override
    `setup()` to load models, and decorate one or more methods with
    `@realtime("/path")`.
    """

    # Identity
    app_id: str = ""

    # Image
    image_ref: str = ""

    # Compute
    compute_type: str = "GPU"
    machine_type: str = "H100 80GB"
    cpu_flavor: str = "cpu3c"
    vcpu_count: int = 2

    # Scaling
    min_concurrency: int = 0
    max_concurrency: int = 4
    keep_alive: int = 300

    # Infrastructure
    container_disk_gb: int = 40
    cloud_type: str = "SECURE"

    def setup(self) -> None:
        """Override to load models. Runs once per pod boot, before serving."""

    @classmethod
    def spawn(
        cls,
        *,
        controller_url: str | None = None,
        api_key: str | None = None,
    ) -> SpawnInfo:
        """Deploy the app and cold-start a pod in one call.

        Returns a SpawnInfo with the app_id and pod_url.
        Loads .env automatically for convenience.
        """
        from dotenv import load_dotenv

        load_dotenv()
        app_id = cls.app_id or cls.__name__.lower()
        deploy(cls, controller_url=controller_url, api_key=api_key)
        pod_url = resolve(app_id, controller_url=controller_url, api_key=api_key)
        return SpawnInfo(app_id=app_id, pod_url=pod_url)


def realtime(path: str):
    """Mark a method as a websocket endpoint mounted at `path`."""

    def decorator(fn):
        fn._lucent_realtime_path = path
        return fn

    return decorator


# ── Controller helpers ────────────────────────────────────────────────

def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or os.environ.get("LUCENT_API_KEY", "")
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def _controller_url(override: str | None = None) -> str:
    url = override or CONTROLLER_URL
    if not url:
        raise RuntimeError(
            "No controller URL. Set LUCENT_CONTROLLER_URL env var "
            "or pass controller_url=."
        )
    return url


def deploy(
    app_cls: type[App],
    *,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Register an App class with the controller.

    Reads all configuration from class attributes. Call this once
    (or on every deploy) — it upserts, so re-running is safe.
    """
    app_id = app_cls.app_id or app_cls.__name__.lower()

    import httpx

    if not app_cls.image_ref:
        raise ValueError(f"{app_cls.__name__}.image_ref is required")

    body = {
        "app_id": app_id,
        "image_ref": app_cls.image_ref,
        "compute_type": app_cls.compute_type,
        "machine_type": app_cls.machine_type,
        "cpu_flavor": app_cls.cpu_flavor,
        "vcpu_count": app_cls.vcpu_count,
        "min_concurrency": app_cls.min_concurrency,
        "max_concurrency": app_cls.max_concurrency,
        "keep_alive_sec": app_cls.keep_alive,
        "container_disk_gb": app_cls.container_disk_gb,
        "cloud_type": app_cls.cloud_type,
    }

    url = _controller_url(controller_url)
    with httpx.Client(timeout=30.0) as c:
        resp = c.post(f"{url}/apps", json=body, headers=_headers(api_key))
    if resp.status_code != 200:
        raise RuntimeError(f"deploy failed ({resp.status_code}): {resp.text}")

    print(f"deployed {app_id} -> {app_cls.image_ref}")
    print(f"resolve with: ls.resolve({app_id!r})")


def resolve(
    app_id: str,
    *,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Get a pod URL for the given app from the controller.

    Cold-starts a pod if none are warm. Returns the pod's base URL
    (e.g. "https://{pod}-8000.proxy.runpod.net").
    """
    import httpx

    url = _controller_url(controller_url)
    with httpx.Client(timeout=180.0) as c:
        resp = c.get(
            f"{url}/resolve",
            params={"app_id": app_id},
            headers=_headers(api_key),
        )
    if resp.status_code != 200:
        raise RuntimeError(f"resolve failed ({resp.status_code}): {resp.text}")
    return resp.json()["pod_url"]


@asynccontextmanager
async def connect(app_id: str, path: str = "/realtime", **resolve_kwargs):
    """Resolve a pod and open a websocket in one call.

    Usage:
        async with ls.connect("echo", "/realtime") as ws:
            await ws.send("hello")
            print(await ws.recv())
    """
    import websockets

    pod_url = resolve(app_id, **resolve_kwargs)
    ws_url = pod_url.replace("https://", "wss://") + path

    async with websockets.connect(ws_url) as ws:
        yield ws


__all__ = ["App", "SpawnInfo", "realtime", "deploy", "resolve", "connect"]
