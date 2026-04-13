"""lucent-spawn: spawn a pod on RunPod, wait for /health, print the public URL.

This is the manual stand-in for the Layer 3 scheduler. Once the control
plane exists, clients hit /resolve and this CLI stops being on the
happy path — but it stays as a dev/debug tool.

Usage:
  lucent-spawn <image>
  lucent-spawn <image> --gpu-type "NVIDIA GeForce RTX 4090" --name echo-test
  lucent-spawn <image> --app-id echo --port 8000
  lucent-spawn <image> --no-wait     # print id + URL and return immediately
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from ..runpod_client import (
    PodSpec,
    RunpodError,
    create_pod,
    pod_proxy_url,
    wait_until_ready,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lucent-spawn",
        description="Spawn a lucent-serverless pod on RunPod and print its proxy URL.",
    )
    p.add_argument("image", help="container image ref, e.g. ghcr.io/you/lucent-echo:dev")
    p.add_argument("--name", default="lucent-spawn", help="pod name (for the RunPod console)")
    p.add_argument(
        "--compute-type",
        default="GPU",
        choices=["GPU", "CPU"],
        help="GPU pods need --gpu-type; CPU pods need --cpu-flavor",
    )
    p.add_argument(
        "--gpu-type",
        default="NVIDIA GeForce RTX 4090",
        help="RunPod gpuTypeId, e.g. 'NVIDIA H100 80GB HBM3' (ignored when --compute-type CPU)",
    )
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument(
        "--cpu-flavor",
        default="cpu3c",
        help="RunPod cpuFlavorId, e.g. 'cpu3c' (ignored when --compute-type GPU)",
    )
    p.add_argument("--vcpu-count", type=int, default=2)
    p.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--container-disk-gb", type=int, default=40)
    p.add_argument("--volume-gb", type=int, default=0)
    p.add_argument("--port", type=int, default=8000, help="container port exposed via RunPod's HTTP proxy")
    p.add_argument("--app-id", default=None, help="value for the LUCENT_APP_ID env var in the pod")
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra env var to set on the pod (repeatable)",
    )
    p.add_argument("--no-wait", action="store_true", help="skip /health polling and return immediately")
    p.add_argument("--timeout", type=float, default=300.0, help="seconds to wait for /health")
    return p


def _parse_env(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--env {pair!r}: expected KEY=VALUE")
        k, v = pair.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()

    env = _parse_env(args.env)
    if args.app_id:
        env.setdefault("LUCENT_APP_ID", args.app_id)

    spec = PodSpec(
        name=args.name,
        image=args.image,
        compute_type=args.compute_type,
        gpu_type_ids=[args.gpu_type] if args.compute_type == "GPU" else [],
        gpu_count=args.gpu_count,
        cpu_flavor_ids=[args.cpu_flavor] if args.compute_type == "CPU" else [],
        vcpu_count=args.vcpu_count,
        cloud_type=args.cloud_type,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        ports=[f"{args.port}/http"],
        env=env,
    )

    hw = args.gpu_type if args.compute_type == "GPU" else f"CPU {args.cpu_flavor}"
    print(f"creating pod ({args.image} on {hw})...", file=sys.stderr)
    try:
        pod = create_pod(spec)
    except RunpodError as e:
        print(f"error creating pod: {e}", file=sys.stderr)
        return 2

    pod_id = pod["id"]
    url = pod_proxy_url(pod_id, args.port)
    print(f"pod_id:    {pod_id}", file=sys.stderr)
    print(f"proxy_url: {url}", file=sys.stderr)

    if args.no_wait:
        print(url)
        return 0

    print(f"waiting for {url}/health (timeout {args.timeout:.0f}s)...", file=sys.stderr)
    try:
        wait_until_ready(pod_id, port=args.port, timeout=args.timeout)
    except (TimeoutError, RunpodError) as e:
        print(f"pod did not become ready: {e}", file=sys.stderr)
        print(f"(pod is still running — terminate with: lucent-terminate {pod_id})", file=sys.stderr)
        return 3

    print("pod ready", file=sys.stderr)
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
