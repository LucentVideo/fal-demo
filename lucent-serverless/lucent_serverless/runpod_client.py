"""REST client for the RunPod v1 API. Transport-only, no business logic.

Endpoints used:
  POST   /v1/pods           — CreatePod
  GET    /v1/pods/{podId}   — GetPod
  DELETE /v1/pods/{podId}   — DeletePod (terminate; releases volume, stops billing)

Auth is `Authorization: Bearer <RUNPOD_API_KEY>`. The library reads the key
from the process environment; CLI entry points are responsible for loading
.env before calling into here.

The RunPod proxy URL `https://{pod_id}-{port}.proxy.runpod.net` is a
convention, not a field returned by the API. It only works when the port
is declared with the `/http` protocol at create time — see `PodSpec.ports`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_TIMEOUT = 30.0

# Terminal statuses per the OpenAPI schema: RUNNING | EXITED | TERMINATED
TERMINAL_STATUSES = frozenset({"EXITED", "TERMINATED"})


class RunpodError(RuntimeError):
    """Any non-2xx response from RunPod's REST API, or a pod that reached a
    terminal status before it was ready."""

    def __init__(self, status: int, body: Any):
        super().__init__(f"runpod error {status}: {body}")
        self.status = status
        self.body = body


@dataclass
class PodSpec:
    """Minimal spec to create a pod. A lucent_serverless.App gets translated
    to this by whoever is calling create_pod (a CLI today, the scheduler
    tomorrow).

    compute_type switches between GPU and CPU pods. For CPU pods, set
    cpu_flavor_ids (e.g. ["cpu3c"]) and vcpu_count; gpu_type_ids/gpu_count
    are ignored by the API per the OpenAPI spec.
    """

    name: str
    image: str
    compute_type: str = "GPU"
    gpu_type_ids: list[str] = field(default_factory=list)
    gpu_count: int = 1
    cpu_flavor_ids: list[str] = field(default_factory=list)
    vcpu_count: int = 2
    container_disk_gb: int = 40
    volume_gb: int = 0
    ports: list[str] = field(default_factory=lambda: ["8000/http"])
    env: dict[str, str] = field(default_factory=dict)
    cloud_type: str = "SECURE"
    interruptible: bool = False


def _require_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY is not set. Put it in your environment or .env "
            "before calling runpod_client."
        )
    return key


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=RUNPOD_REST_BASE,
        headers={"Authorization": f"Bearer {_require_api_key()}"},
        timeout=DEFAULT_TIMEOUT,
    )


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RunpodError(resp.status_code, body)


def create_pod(spec: PodSpec) -> dict:
    """POST /v1/pods. Returns the full Pod object from the 201 response.

    The response includes `id` immediately but `portMappings` / `publicIp`
    may still be null until the container finishes booting. Use
    `wait_until_ready` to block on that.
    """
    body: dict[str, Any] = {
        "name": spec.name,
        "imageName": spec.image,
        "computeType": spec.compute_type,
        "cloudType": spec.cloud_type,
        "interruptible": spec.interruptible,
        "containerDiskInGb": spec.container_disk_gb,
        "volumeInGb": spec.volume_gb,
        "ports": spec.ports,
        "env": spec.env,
    }
    if spec.compute_type == "GPU":
        body["gpuTypeIds"] = spec.gpu_type_ids
        body["gpuCount"] = spec.gpu_count
    elif spec.compute_type == "CPU":
        body["cpuFlavorIds"] = spec.cpu_flavor_ids
        body["vcpuCount"] = spec.vcpu_count
    else:
        raise ValueError(f"unknown compute_type {spec.compute_type!r} (expected GPU or CPU)")
    with _client() as c:
        resp = c.post("/pods", json=body)
    _raise_for_status(resp)
    return resp.json()


def get_pod(pod_id: str) -> dict:
    """GET /v1/pods/{podId}."""
    with _client() as c:
        resp = c.get(f"/pods/{pod_id}")
    _raise_for_status(resp)
    return resp.json()


def terminate_pod(pod_id: str) -> None:
    """DELETE /v1/pods/{podId}. Destroys the pod AND its volume. Billing stops."""
    with _client() as c:
        resp = c.delete(f"/pods/{pod_id}")
    _raise_for_status(resp)


def pod_proxy_url(pod_id: str, port: int = 8000) -> str:
    """Construct the RunPod proxy URL for a pod + container port.

    Only valid if the port was declared `/http` in the PodSpec. RunPod
    terminates TLS on its edge and routes to the container port.
    """
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def wait_until_ready(
    pod_id: str,
    *,
    port: int = 8000,
    health_path: str = "/health",
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> dict:
    """Poll until the pod's proxy /health endpoint returns 200.

    Stronger guarantee than "RunPod says RUNNING": we want uvicorn inside
    the container to actually be serving. Returns the pod object at the
    moment it became ready. Raises TimeoutError on timeout and RunpodError
    if the pod reaches a terminal status before readiness.
    """
    url = f"{pod_proxy_url(pod_id, port)}{health_path}"
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None

    with httpx.Client(timeout=5.0) as probe:
        while time.monotonic() < deadline:
            pod = get_pod(pod_id)
            status = pod.get("desiredStatus")
            if status in TERMINAL_STATUSES:
                raise RunpodError(
                    0, f"pod {pod_id} reached terminal status {status} before becoming ready"
                )
            try:
                resp = probe.get(url)
                if resp.status_code == 200:
                    return pod
                last_err = RunpodError(resp.status_code, resp.text)
            except httpx.HTTPError as e:
                # connection refused / DNS not resolving yet while container boots
                last_err = e
            time.sleep(poll_interval)

    raise TimeoutError(
        f"pod {pod_id} did not become ready within {timeout:.0f}s "
        f"(last probe error: {last_err})"
    )
