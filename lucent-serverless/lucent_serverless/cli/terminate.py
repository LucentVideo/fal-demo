"""lucent-terminate: destroy a RunPod pod by id. Stops billing.

Uses DELETE /v1/pods/{podId}, which releases the volume. If you wanted to
keep the volume and just pause, you'd use POST /v1/pods/{podId}/stop —
we intentionally do not expose that because scale-to-zero means "volume
should not exist while idle."
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from ..runpod_client import RunpodError, terminate_pod


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lucent-terminate",
        description="Terminate (destroy) a RunPod pod by id. Releases the volume and stops billing.",
    )
    p.add_argument("pod_id", help="the RunPod pod id to terminate")
    return p


def main() -> int:
    load_dotenv()
    args = _parser().parse_args()

    print(f"terminating {args.pod_id}...", file=sys.stderr)
    try:
        terminate_pod(args.pod_id)
    except RunpodError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
