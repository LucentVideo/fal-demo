"""lucent-serverless: a small serverless platform.

The user-facing surface:

Realtime path:
  - @realtime(path): decorator to mark websocket handlers
  - resolve(app_id): get a pod URL (cold-starts if needed)
  - connect(app_id, path): resolve + open websocket in one call

Job path:
  - @handler: decorator to mark the job handler method
  - submit(app_id, input): enqueue a job, returns job_id
  - poll(job_id): check job status and retrieve result
  - run_sync(app_id, input): submit + block until done

Common:
  - App: base class — subclass it, set attributes, override setup()
  - deploy(AppClass): register your app with the controller
"""

from __future__ import annotations

import io
import os
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path

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
    `setup()` to load models, and decorate methods with `@realtime`
    (for websocket apps) or `@handler` (for job-based apps).
    """

    # Identity
    app_id: str = ""

    # Image
    image_ref: str = ""

    # Mode: "realtime" (websocket) or "job" (queue-based)
    mode: str = "realtime"

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

    # Dependencies — installed at boot (like fal's requirements list)
    requirements: list[str] = []

    # Only tar these top-level entries from source_dir. If empty, ship
    # everything (minus common junk). Analogous to fal's
    # local_python_modules but also accepts files like 'app.py'.
    include: list[str] = []

    def setup(self) -> None:
        """Override to load models. Runs once per pod boot, before serving."""

    @classmethod
    def spawn(
        cls,
        *,
        source_dir: str | Path | None = None,
        env: dict[str, str] | None = None,
        controller_url: str | None = None,
        api_key: str | None = None,
    ) -> SpawnInfo:
        """Deploy the app and cold-start a pod in one call.

        Returns a SpawnInfo with the app_id and pod_url.
        Loads .env automatically for convenience.

        source_dir: directory containing app code to upload. If None,
        inferred from the file where the App subclass is defined.
        env: dict of env vars to inject into every spawned pod.
        """
        import inspect as _inspect

        from dotenv import load_dotenv

        load_dotenv()
        app_id = cls.app_id or cls.__name__.lower()
        deploy(cls, env=env, controller_url=controller_url, api_key=api_key)

        # Upload the app code
        if source_dir is None:
            src_file = _inspect.getfile(cls)
            source_dir = Path(src_file).resolve().parent
        upload_code(
            app_id, source_dir,
            requirements=cls.requirements or None,
            include=cls.include or None,
            controller_url=controller_url, api_key=api_key,
        )

        pod_url = resolve(app_id, controller_url=controller_url, api_key=api_key)
        return SpawnInfo(app_id=app_id, pod_url=pod_url)


def realtime(path: str, *, buffering: int | None = None):
    """Mark a method as a websocket endpoint mounted at `path`.

    If the decorated method's signature uses AsyncIterator type hints
    (fal-compatible style), the runner will automatically wrap the raw
    websocket into async iterators of pydantic models. Otherwise the
    method receives a raw FastAPI WebSocket.
    """

    def decorator(fn):
        fn._lucent_realtime_path = path
        fn._lucent_realtime_buffering = buffering
        return fn

    return decorator


def handler(fn):
    """Mark a method as the job handler for this app."""
    fn._lucent_handler = True
    return fn


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
    env: dict[str, str] | None = None,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Register an App class with the controller.

    Reads all configuration from class attributes. Call this once
    (or on every deploy) — it upserts, so re-running is safe.
    """
    app_id = app_cls.app_id or app_cls.__name__.lower()

    import httpx

    # Default to platform base images when user doesn't specify one
    image_ref = app_cls.image_ref
    if not image_ref:
        if app_cls.compute_type == "CPU":
            image_ref = "raylightdimi/lucent-base-cpu:latest"
        else:
            image_ref = "raylightdimi/lucent-base-gpu:latest"

    body = {
        "app_id": app_id,
        "image_ref": image_ref,
        "compute_type": app_cls.compute_type,
        "machine_type": app_cls.machine_type,
        "cpu_flavor": app_cls.cpu_flavor,
        "vcpu_count": app_cls.vcpu_count,
        "min_concurrency": app_cls.min_concurrency,
        "max_concurrency": app_cls.max_concurrency,
        "keep_alive_sec": app_cls.keep_alive,
        "container_disk_gb": app_cls.container_disk_gb,
        "cloud_type": app_cls.cloud_type,
        "mode": app_cls.mode,
        "env": env or {},
    }

    url = _controller_url(controller_url)
    with httpx.Client(timeout=30.0) as c:
        resp = c.post(f"{url}/apps", json=body, headers=_headers(api_key))
    if resp.status_code != 200:
        raise RuntimeError(f"deploy failed ({resp.status_code}): {resp.text}")

    print(f"deployed {app_id} -> {image_ref}")
    print(f"resolve with: ls.resolve({app_id!r})")


def _tar_directory(
    directory: Path,
    requirements: list[str] | None = None,
    include: list[str] | None = None,
) -> bytes:
    """Create a tar.gz of the directory contents (not the directory itself).

    If requirements is provided, a requirements.txt is generated and
    injected into the tarball (overriding any existing one).
    If include is provided, only those top-level names from the
    directory are packaged (instead of iterating the whole dir).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if include:
            items = [directory / name for name in include]
            missing = [p for p in items if not p.exists()]
            if missing:
                raise ValueError(
                    f"include paths not found in {directory}: "
                    f"{[p.name for p in missing]}"
                )
        else:
            items = sorted(directory.iterdir())
        for item in items:
            # Skip common junk
            if item.name in {"__pycache__", ".git", ".venv", "node_modules", ".env"}:
                continue
            # Skip existing requirements.txt if we're generating one
            if item.name == "requirements.txt" and requirements:
                continue
            tar.add(item, arcname=item.name)

        # Inject generated requirements.txt from class attribute
        if requirements:
            reqs_content = "\n".join(requirements) + "\n"
            reqs_bytes = reqs_content.encode()
            info = tarfile.TarInfo(name="requirements.txt")
            info.size = len(reqs_bytes)
            tar.addfile(info, io.BytesIO(reqs_bytes))

    return buf.getvalue()


def upload_code(
    app_id: str,
    source_dir: str | Path,
    *,
    requirements: list[str] | None = None,
    include: list[str] | None = None,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Tar up a directory and upload it to the controller as app code."""
    import httpx

    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError(f"{source} is not a directory")

    data = _tar_directory(source, requirements=requirements, include=include)
    url = _controller_url(controller_url)
    with httpx.Client(timeout=60.0) as c:
        resp = c.put(
            f"{url}/apps/{app_id}/code",
            content=data,
            headers={
                **_headers(api_key),
                "Content-Type": "application/gzip",
            },
        )
    if resp.status_code != 200:
        raise RuntimeError(f"code upload failed ({resp.status_code}): {resp.text}")

    print(f"uploaded {len(data)} bytes of code for {app_id}")


def resolve(
    app_id: str,
    *,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Get a ready pod URL for the given app from the controller.

    The controller's /resolve endpoint is non-blocking: it returns
    status="pending" while the pod is still cold-starting, and
    status="ready" once the scheduler has promoted it. We poll here
    until ready or timeout.
    """
    import time

    import httpx

    url = _controller_url(controller_url)
    headers = _headers(api_key)
    deadline = time.time() + timeout
    last_status = None

    with httpx.Client(timeout=15.0) as c:
        while True:
            resp = c.get(
                f"{url}/resolve",
                params={"app_id": app_id},
                headers=headers,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"resolve failed ({resp.status_code}): {resp.text}"
                )
            body = resp.json()
            status = body.get("status", "ready")
            if status == "ready":
                return body["pod_url"]
            if status != last_status:
                print(f"resolve: pod {body.get('pod_id')} status={status}, waiting...")
                last_status = status
            if time.time() > deadline:
                raise TimeoutError(
                    f"pod for {app_id!r} did not become ready within {timeout:.0f}s"
                )
            time.sleep(poll_interval)


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


# ── Job client helpers ────────────────────────────────────────────────

def submit(
    app_id: str,
    input: dict,
    *,
    webhook_url: str | None = None,
    ttl_sec: int = 300,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Enqueue a job for a job-mode app. Returns the job_id."""
    import httpx

    url = _controller_url(controller_url)
    body = {"app_id": app_id, "input": input, "ttl_sec": ttl_sec}
    if webhook_url:
        body["webhook_url"] = webhook_url

    with httpx.Client(timeout=30.0) as c:
        resp = c.post(f"{url}/run", json=body, headers=_headers(api_key))
    if resp.status_code != 200:
        raise RuntimeError(f"submit failed ({resp.status_code}): {resp.text}")
    return resp.json()["job_id"]


def poll(
    job_id: str,
    *,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Check the status of a job. Returns the full job dict."""
    import httpx

    url = _controller_url(controller_url)
    with httpx.Client(timeout=30.0) as c:
        resp = c.get(f"{url}/status/{job_id}", headers=_headers(api_key))
    if resp.status_code != 200:
        raise RuntimeError(f"poll failed ({resp.status_code}): {resp.text}")
    return resp.json()


def run_sync(
    app_id: str,
    input: dict,
    *,
    timeout: int = 120,
    controller_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Submit a job and block until it completes. Returns the result."""
    import httpx

    url = _controller_url(controller_url)
    body = {"app_id": app_id, "input": input, "ttl_sec": timeout}

    with httpx.Client(timeout=float(timeout) + 10) as c:
        resp = c.post(f"{url}/runsync", json=body, headers=_headers(api_key))
    if resp.status_code == 408:
        raise TimeoutError(f"job did not complete within {timeout}s")
    if resp.status_code != 200:
        raise RuntimeError(f"run_sync failed ({resp.status_code}): {resp.text}")
    return resp.json()


__all__ = [
    "App", "SpawnInfo",
    "realtime", "handler",
    "deploy", "upload_code",
    "resolve", "connect",
    "submit", "poll", "run_sync",
]