"""Idle watchdog: terminate the pod when nobody has been around for keep_alive.

os._exit (not sys.exit) so we bypass cleanup handlers that might hang on a
half-shut CUDA context. The control plane's reaper is a backup; this is
the first line of defense against runaway billing.
"""

import asyncio
import logging
import os
import time

from . import state

log = logging.getLogger(__name__)


async def run(keep_alive: int, check_interval: int = 30) -> None:
    while True:
        await asyncio.sleep(check_interval)
        snap = state.snapshot()
        if snap["active_connections"] > 0:
            continue
        idle_for = time.time() - snap["last_activity_at"]
        if idle_for < keep_alive:
            continue
        log.info("idle %.0fs >= keep_alive %ds, exiting", idle_for, keep_alive)
        os._exit(0)
