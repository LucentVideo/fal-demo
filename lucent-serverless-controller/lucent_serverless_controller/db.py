"""SQLite database for the control plane.

Two tables: apps and pods. Schema follows notes/myfal_plan.md section 5,
adapted for what we actually learned building Layers 1-2 (CPU support,
no in-pod watchdog, RunPod REST ids as primary keys).

Thread safety: each thread gets its own connection via threading.local().
WAL mode allows concurrent reads from the scheduler, reaper, and FastAPI
handler threads without blocking.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path("lucent_controller.db")

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH))
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS apps (
            app_id            TEXT PRIMARY KEY,
            image_ref         TEXT NOT NULL,
            compute_type      TEXT NOT NULL DEFAULT 'GPU',
            machine_type      TEXT NOT NULL DEFAULT 'NVIDIA GeForce RTX 4090',
            cpu_flavor        TEXT DEFAULT 'cpu3c',
            vcpu_count        INTEGER DEFAULT 2,
            min_concurrency   INTEGER NOT NULL DEFAULT 0,
            max_concurrency   INTEGER NOT NULL DEFAULT 4,
            keep_alive_sec    INTEGER NOT NULL DEFAULT 300,
            container_disk_gb INTEGER NOT NULL DEFAULT 40,
            cloud_type        TEXT NOT NULL DEFAULT 'SECURE',
            env               TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS pods (
            pod_id              TEXT PRIMARY KEY,
            app_id              TEXT NOT NULL REFERENCES apps(app_id),
            url                 TEXT,
            status              TEXT NOT NULL DEFAULT 'pending',
            active_connections  INTEGER NOT NULL DEFAULT 0,
            last_activity_at    INTEGER NOT NULL,
            started_at          INTEGER NOT NULL,
            last_health_poll    INTEGER,
            failed_polls        INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_pods_app_status
            ON pods(app_id, status);
    """)
    conn.commit()


# ── App CRUD ──────────────────────────────────────────────────────────

def get_app(app_id: str) -> sqlite3.Row | None:
    return _conn().execute(
        "SELECT * FROM apps WHERE app_id = ?", (app_id,)
    ).fetchone()


def list_apps() -> list[sqlite3.Row]:
    return _conn().execute("SELECT * FROM apps").fetchall()


def upsert_app(
    app_id: str,
    image_ref: str,
    *,
    compute_type: str = "GPU",
    machine_type: str = "NVIDIA GeForce RTX 4090",
    cpu_flavor: str = "cpu3c",
    vcpu_count: int = 2,
    min_concurrency: int = 0,
    max_concurrency: int = 4,
    keep_alive_sec: int = 300,
    container_disk_gb: int = 40,
    cloud_type: str = "SECURE",
    env: dict | None = None,
) -> None:
    _conn().execute(
        """INSERT INTO apps (
               app_id, image_ref, compute_type, machine_type, cpu_flavor,
               vcpu_count, min_concurrency, max_concurrency, keep_alive_sec,
               container_disk_gb, cloud_type, env
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(app_id) DO UPDATE SET
               image_ref=excluded.image_ref,
               compute_type=excluded.compute_type,
               machine_type=excluded.machine_type,
               cpu_flavor=excluded.cpu_flavor,
               vcpu_count=excluded.vcpu_count,
               min_concurrency=excluded.min_concurrency,
               max_concurrency=excluded.max_concurrency,
               keep_alive_sec=excluded.keep_alive_sec,
               container_disk_gb=excluded.container_disk_gb,
               cloud_type=excluded.cloud_type,
               env=excluded.env
        """,
        (app_id, image_ref, compute_type, machine_type, cpu_flavor,
         vcpu_count, min_concurrency, max_concurrency, keep_alive_sec,
         container_disk_gb, cloud_type, json.dumps(env or {})),
    )
    _conn().commit()


def delete_app(app_id: str) -> bool:
    cur = _conn().execute("DELETE FROM apps WHERE app_id = ?", (app_id,))
    _conn().commit()
    return cur.rowcount > 0


# ── Pod CRUD ──────────────────────────────────────────────────────────

def insert_pod(pod_id: str, app_id: str, url: str | None = None) -> None:
    now = int(time.time())
    _conn().execute(
        """INSERT INTO pods
           (pod_id, app_id, url, status, active_connections,
            last_activity_at, started_at)
           VALUES (?, ?, ?, 'pending', 0, ?, ?)""",
        (pod_id, app_id, url, now, now),
    )
    _conn().commit()


def get_pod(pod_id: str) -> sqlite3.Row | None:
    return _conn().execute(
        "SELECT * FROM pods WHERE pod_id = ?", (pod_id,)
    ).fetchone()


def pods_by_app_and_status(
    app_id: str, statuses: list[str]
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in statuses)
    return _conn().execute(
        f"""SELECT * FROM pods
            WHERE app_id = ? AND status IN ({placeholders})
            ORDER BY active_connections DESC""",
        [app_id, *statuses],
    ).fetchall()


def live_pods() -> list[sqlite3.Row]:
    return _conn().execute(
        "SELECT * FROM pods WHERE status IN ('pending', 'ready')"
    ).fetchall()


def ready_idle_pods() -> list[sqlite3.Row]:
    return _conn().execute(
        "SELECT * FROM pods WHERE status = 'ready' AND active_connections = 0"
    ).fetchall()


def update_pod_health(
    pod_id: str, active_connections: int, last_activity_at: int
) -> None:
    _conn().execute(
        """UPDATE pods SET
               active_connections = ?,
               last_activity_at = ?,
               last_health_poll = ?,
               failed_polls = 0
           WHERE pod_id = ?""",
        (active_connections, last_activity_at, int(time.time()), pod_id),
    )
    _conn().commit()


def promote_pod(pod_id: str, url: str) -> None:
    _conn().execute(
        "UPDATE pods SET status = 'ready', url = ? WHERE pod_id = ?",
        (url, pod_id),
    )
    _conn().commit()


def increment_failed_polls(pod_id: str) -> int:
    conn = _conn()
    conn.execute(
        "UPDATE pods SET failed_polls = failed_polls + 1 WHERE pod_id = ?",
        (pod_id,),
    )
    conn.commit()
    row = conn.execute(
        "SELECT failed_polls FROM pods WHERE pod_id = ?", (pod_id,)
    ).fetchone()
    return row["failed_polls"] if row else 0


def set_pod_status(pod_id: str, status: str) -> None:
    _conn().execute(
        "UPDATE pods SET status = ? WHERE pod_id = ?", (status, pod_id)
    )
    _conn().commit()
