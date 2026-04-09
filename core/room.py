"""Multiplayer room: peer tracking, frame broadcasting, and the shared
processing loop that feeds all connected viewers.

The Room holds a single FrameBroadcaster. One background asyncio task
reads frames from the active user's inbound WebRTC track, runs perception
inference, and publishes processed VideoFrames to all subscribers.  Each
connected peer gets a BroadcastTrack whose ``recv()`` returns the latest
published frame.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("room")
log.setLevel(logging.DEBUG)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    log.addHandler(_h)


@dataclass
class PeerState:
    """Mutable state for a single connected peer."""

    peer_id: str
    username: str
    video_track: Any = None
    outgoing_queue: asyncio.Queue | None = None


class FrameBroadcaster:
    """Publish-subscribe fan-out for processed video frames.

    Each subscriber gets its own ``asyncio.Event`` that is set whenever a
    new frame is published. Late subscribers always see the latest frame.
    """

    def __init__(self) -> None:
        self._frame: Any = None
        self._frame_id: int = 0
        self._subscribers: list[BroadcastSubscriber] = []

    async def publish(self, frame: Any) -> None:
        self._frame = frame
        self._frame_id += 1
        for sub in self._subscribers:
            sub._notify()

    def subscribe(self) -> BroadcastSubscriber:
        sub = BroadcastSubscriber(self)
        self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: BroadcastSubscriber) -> None:
        sub.close()
        with suppress(ValueError):
            self._subscribers.remove(sub)


class BroadcastSubscriber:
    """Receives the latest frame from a FrameBroadcaster."""

    def __init__(self, broadcaster: FrameBroadcaster) -> None:
        self._broadcaster = broadcaster
        self._last_seen_id: int = 0
        self._event = asyncio.Event()
        self._closed = False

    def _notify(self) -> None:
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def recv(self) -> Any:
        while self._last_seen_id >= self._broadcaster._frame_id:
            if self._closed:
                raise Exception("subscriber closed")
            self._event.clear()
            await self._event.wait()
        if self._closed:
            raise Exception("subscriber closed")
        self._last_seen_id = self._broadcaster._frame_id
        return self._broadcaster._frame


def create_broadcast_track(subscriber: BroadcastSubscriber):
    """Factory: creates an aiortc MediaStreamTrack fed by a subscriber."""
    from aiortc.mediastreams import MediaStreamTrack

    class BroadcastTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._sub = subscriber

        async def recv(self):
            return await self._sub.recv()

        def stop(self) -> None:
            super().stop()
            self._sub.close()

    return BroadcastTrack()


class Room:
    """Single multiplayer room shared across all WebSocket handlers."""

    def __init__(self, *, yolo: Any, depth: Any, seg: Any) -> None:
        self.peers: dict[str, PeerState] = {}
        self._peer_order: list[str] = []
        self.active_peer_id: str | None = None
        self.active_layer: str = "detection"
        self.broadcaster = FrameBroadcaster()

        self._yolo = yolo
        self._depth = depth
        self._seg = seg

        self._processing_task: asyncio.Task | None = None
        self._stopped = False
        self._frame_count = 0
        self._timing_every_n = 5

        self._drain_tasks: dict[str, asyncio.Task] = {}

    # ---- peer management ------------------------------------------------

    def add_peer(
        self, peer_id: str, username: str, outgoing_queue: asyncio.Queue
    ) -> None:
        self.peers[peer_id] = PeerState(
            peer_id=peer_id,
            username=username,
            outgoing_queue=outgoing_queue,
        )
        self._peer_order.append(peer_id)
        auto_active = self.active_peer_id is None
        if auto_active:
            self.active_peer_id = peer_id
        log.info(f"room: add_peer {peer_id} ({username}), auto_active={auto_active}, peers={list(self.peers.keys())}")
        self._ensure_processing()

    def remove_peer(self, peer_id: str) -> None:
        was_active = self.active_peer_id == peer_id
        self._stop_drain(peer_id)
        self.peers.pop(peer_id, None)
        with suppress(ValueError):
            self._peer_order.remove(peer_id)
        if was_active:
            new_active = self._peer_order[0] if self._peer_order else None
            self.active_peer_id = new_active
            log.info(f"room: remove_peer {peer_id} (was active), promoted={new_active}")
            if new_active:
                self._stop_drain(new_active)
        else:
            log.info(f"room: remove_peer {peer_id} (was spectator)")
        log.info(f"room: remaining peers={list(self.peers.keys())}, active={self.active_peer_id}")
        if not self.peers:
            self._stop_processing()

    def set_peer_track(self, peer_id: str, track: Any) -> None:
        peer = self.peers.get(peer_id)
        if peer is not None:
            peer.video_track = track
            is_active = peer_id == self.active_peer_id
            log.info(f"room: set_peer_track {peer_id}, is_active={is_active}, track={track}")
            if not is_active:
                self._start_drain(peer_id)
        else:
            log.info(f"room: set_peer_track {peer_id} IGNORED — peer not registered")

    def set_active(self, peer_id: str) -> None:
        if peer_id not in self.peers:
            log.info(f"room: set_active {peer_id} IGNORED — not in peers")
            return
        old_active = self.active_peer_id
        self.active_peer_id = peer_id
        old_track = self.peers[old_active].video_track if old_active and old_active in self.peers else None
        new_track = self.peers[peer_id].video_track
        log.info(f"room: set_active {old_active} -> {peer_id}, old_has_track={old_track is not None}, new_has_track={new_track is not None}")
        self._stop_drain(peer_id)
        if old_active and old_active != peer_id and old_active in self.peers:
            self._start_drain(old_active)

    # ---- state & broadcasting -------------------------------------------

    def get_state_dict(self) -> dict:
        return {
            "type": "room_state",
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "username": p.username,
                    "is_active": p.peer_id == self.active_peer_id,
                    "has_video": p.video_track is not None,
                }
                for p in (
                    self.peers[pid]
                    for pid in self._peer_order
                    if pid in self.peers
                )
            ],
            "active_peer_id": self.active_peer_id,
        }

    def _enqueue_to_all(self, msg: Any) -> None:
        for peer in self.peers.values():
            if peer.outgoing_queue is not None:
                with suppress(Exception):
                    peer.outgoing_queue.put_nowait(msg)

    async def broadcast_room_state(self) -> None:
        self._enqueue_to_all(self.get_state_dict())

    # ---- drain non-active tracks ----------------------------------------
    # Without draining, aiortc buffers decoded frames on tracks nobody
    # reads from.  When control switches, the processing loop would have
    # to chew through the entire backlog before reaching real-time,
    # which looks like a freeze.

    def _start_drain(self, peer_id: str) -> None:
        if peer_id in self._drain_tasks and not self._drain_tasks[peer_id].done():
            log.info(f"room: _start_drain {peer_id} — already draining")
            return
        peer = self.peers.get(peer_id)
        if peer and peer.video_track:
            log.info(f"room: _start_drain {peer_id} — launching drain task")
            self._drain_tasks[peer_id] = asyncio.ensure_future(
                self._drain_loop(peer_id)
            )
        else:
            log.info(f"room: _start_drain {peer_id} — skipped (no peer or no track)")

    def _stop_drain(self, peer_id: str) -> None:
        task = self._drain_tasks.pop(peer_id, None)
        if task and not task.done():
            task.cancel()
            log.info(f"room: _stop_drain {peer_id} — cancelled drain task")
        else:
            log.info(f"room: _stop_drain {peer_id} — nothing to stop")

    async def _drain_loop(self, peer_id: str) -> None:
        """Consume and discard frames from a non-active peer's track."""
        log.info(f"room: drain_loop ENTER {peer_id}")
        drained = 0
        try:
            while True:
                peer = self.peers.get(peer_id)
                if peer is None or peer.video_track is None:
                    log.info(f"room: drain_loop {peer_id} — peer/track gone, exiting")
                    break
                if peer_id == self.active_peer_id:
                    log.info(f"room: drain_loop {peer_id} — became active, exiting")
                    break
                try:
                    await asyncio.wait_for(peer.video_track.recv(), timeout=1.0)
                    drained += 1
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            log.info(f"room: drain_loop {peer_id} — cancelled after draining {drained} frames")
        except Exception as exc:
            log.info(f"room: drain_loop {peer_id} — error: {exc}")
        log.info(f"room: drain_loop EXIT {peer_id}, total drained={drained}")

    # ---- processing loop ------------------------------------------------

    def _get_active_source_track(self) -> Any | None:
        if self.active_peer_id is None:
            return None
        peer = self.peers.get(self.active_peer_id)
        return peer.video_track if peer else None

    def _ensure_processing(self) -> None:
        if self._processing_task is None or self._processing_task.done():
            self._stopped = False
            log.info("_ensure_processing — starting processing loop")
            self._processing_task = asyncio.ensure_future(
                self._processing_loop()
            )
        else:
            log.info("_ensure_processing — already running")

    def _stop_processing(self) -> None:
        self._stopped = True
        if self._processing_task is not None:
            self._processing_task.cancel()
            self._processing_task = None

    async def _processing_loop(self) -> None:
        import time as _time
        from fractions import Fraction

        import numpy as np
        from av import VideoFrame

        from core.compose import compose_frame

        out_pts = 0
        prev_active_id = None
        loop_iter = 0

        log.info("processing_loop STARTED")

        while not self._stopped:
            loop_iter += 1
            source_track = self._get_active_source_track()
            current_active = self.active_peer_id

            if current_active != prev_active_id:
                log.info(f"room: processing_loop source changed: {prev_active_id} -> {current_active}, has_track={source_track is not None}")
                prev_active_id = current_active

            if source_track is None:
                if loop_iter % 30 == 1:
                    log.info(f"room: processing_loop idle (no source track), active_peer={self.active_peer_id}")
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = VideoFrame.from_ndarray(black, format="bgr24")
                frame.pts = out_pts
                frame.time_base = Fraction(1, 30)
                out_pts += 1
                await self.broadcaster.publish(frame)
                await asyncio.sleep(0.1)
                continue

            try:
                raw_frame = await asyncio.wait_for(
                    source_track.recv(), timeout=1.0
                )
            except asyncio.TimeoutError:
                log.info(f"room: processing_loop recv TIMEOUT from {current_active}")
                continue
            except Exception as exc:
                log.info(f"room: processing_loop recv ERROR from {current_active}: {exc}")
                await asyncio.sleep(0.033)
                continue

            img = raw_frame.to_ndarray(format="bgr24")
            layer = self.active_layer

            yolo_out = depth_out = seg_out = None
            yolo_ms = depth_ms = seg_ms = None

            try:
                t_start = _time.perf_counter()

                if layer in ("detection", "composite"):
                    s = _time.perf_counter()
                    yolo_out = self._yolo(img)
                    yolo_ms = (_time.perf_counter() - s) * 1000.0

                if layer in ("depth", "composite"):
                    s = _time.perf_counter()
                    depth_out = self._depth(img)
                    depth_ms = (_time.perf_counter() - s) * 1000.0

                if layer in ("segmentation", "composite"):
                    s = _time.perf_counter()
                    seg_out = self._seg(img)
                    seg_ms = (_time.perf_counter() - s) * 1000.0

                total_ms = (_time.perf_counter() - t_start) * 1000.0

                annotated = compose_frame(
                    img,
                    layer=layer,
                    yolo=yolo_out,
                    depth=depth_out,
                    seg=seg_out,
                )
            except Exception as exc:
                log.info(f"room: processing error: {exc}")
                annotated = img
                total_ms = 0.0

            new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
            new_frame.pts = out_pts
            new_frame.time_base = Fraction(1, 30)
            out_pts += 1
            await self.broadcaster.publish(new_frame)

            self._frame_count += 1
            if self._frame_count % 30 == 1:
                log.info(f"room: processing_loop frame #{self._frame_count} from {current_active}, layer={layer}, {total_ms:.0f}ms, subscribers={len(self.broadcaster._subscribers)}")
            if self._frame_count % self._timing_every_n == 0:
                self._enqueue_to_all(
                    {
                        "type": "timing",
                        "layer": layer,
                        "yolo_ms": yolo_ms,
                        "depth_ms": depth_ms,
                        "seg_ms": seg_ms,
                        "total_ms": total_ms,
                    }
                )
