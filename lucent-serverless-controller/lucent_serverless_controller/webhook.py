"""Fire-and-forget webhook delivery with retries.

Runs the POST in a daemon thread so the caller (jobs.py) is not blocked.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_SEC = 2.0


def deliver_webhook(url: str, payload: dict) -> None:
    """Send *payload* to *url* in a background thread."""
    t = threading.Thread(
        target=_deliver, args=(url, payload), daemon=True, name="webhook"
    )
    t.start()


def _deliver(url: str, payload: dict) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=10.0) as c:
                resp = c.post(url, json=payload)
            if resp.status_code < 400:
                log.info("webhook delivered to %s (attempt %d)", url, attempt)
                return
            log.warning(
                "webhook to %s returned %d (attempt %d/%d)",
                url, resp.status_code, attempt, MAX_ATTEMPTS,
            )
        except httpx.HTTPError as e:
            log.warning(
                "webhook to %s failed (attempt %d/%d): %s",
                url, attempt, MAX_ATTEMPTS, e,
            )
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_SEC * attempt)

    log.error("webhook to %s exhausted all %d attempts", url, MAX_ATTEMPTS)
