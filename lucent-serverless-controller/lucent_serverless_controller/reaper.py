"""Reaper loop: terminate pods that have been idle longer than keep_alive.

This is the only thing keeping the bill from running away. The worker
has no in-pod self-kill (see §4.6 in the plan); the reaper calls
DELETE /pods/{id} from outside.

Runs in its own thread, started by the lifespan in main.py.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from lucent_serverless.runpod_client import RunpodError, terminate_pod

from . import db

log = logging.getLogger(__name__)

REAP_INTERVAL = 30


def _fresh_health(url: str) -> dict | None:
    """Direct /health check on the pod — bypasses stale DB cache."""
    try:
        with httpx.Client(timeout=5.0) as c:
            resp = c.get(f"{url}/health")
        if resp.status_code == 200:
            return resp.json()
    except httpx.HTTPError:
        pass
    return None


def _is_pod_busy(health: dict | None, app_mode: str) -> bool:
    """Check whether a pod is actively doing work, based on app mode."""
    if health is None:
        return False
    if app_mode == "job":
        return not health.get("idle", True)
    return health.get("active_connections", 0) > 0


def _tick() -> None:
    for pod in db.ready_idle_pods():
        app = db.get_app(pod["app_id"])
        if not app:
            continue
        idle_for = time.time() - pod["last_activity_at"]
        if idle_for < app["keep_alive_sec"]:
            continue

        # DB says idle — but the DB is up to 5s stale.
        # Re-check the pod directly before killing it.
        health = _fresh_health(pod["url"])
        if _is_pod_busy(health, app["mode"]):
            log.debug("pod %s looked idle in DB but is busy, skipping", pod["pod_id"])
            continue

        log.info(
            "reaping pod %s (app=%s, idle=%.0fs > keep_alive=%ds)",
            pod["pod_id"], pod["app_id"], idle_for, app["keep_alive_sec"],
        )
        db.set_pod_status(pod["pod_id"], "draining")
        db.set_pod_history_status(pod["pod_id"], "draining")
        try:
            terminate_pod(pod["pod_id"])
        except RunpodError as e:
            log.warning("terminate failed for pod %s: %s", pod["pod_id"], e)
        db.set_pod_status(pod["pod_id"], "dead")
        db.set_pod_history_status(pod["pod_id"], "terminated")


def reaper_loop(stop: threading.Event) -> None:
    log.info("reaper loop started (interval=%ds)", REAP_INTERVAL)
    while not stop.is_set():
        try:
            _tick()
        except Exception:
            log.exception("reaper tick failed")
        stop.wait(REAP_INTERVAL)
    log.info("reaper loop stopped")
