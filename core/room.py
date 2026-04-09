"""Multiplayer room: peer tracking, frame broadcasting, and the shared
processing loop that feeds all connected viewers.

The Room holds a single FrameBroadcaster. One background asyncio task
reads frames from ALL connected peers' inbound WebRTC tracks, runs YOLO
detection on each, composes them into a labelled grid image, and publishes
the composite VideoFrame to all subscribers. Each connected peer gets a
BroadcastTrack whose ``recv()`` returns the latest published frame.
"""

from __future__ import annotations

import asyncio
import logging
import math
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


GRID_W = 640
GRID_H = 480


class Room:
    """Single multiplayer room shared across all WebSocket handlers."""

    def __init__(self, *, yolo: Any, depth: Any, seg: Any, face_swapper: Any = None) -> None:
        self.peers: dict[str, PeerState] = {}
        self._peer_order: list[str] = []
        self.active_layer: str = "detection"
        self.broadcaster = FrameBroadcaster()

        self._yolo = yolo
        self._depth = depth
        self._seg = seg
        self.face_swapper = face_swapper

        self._processing_task: asyncio.Task | None = None
        self._stopped = False
        self._frame_count = 0
        self._timing_every_n = 5

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
        log.info(f"add_peer {peer_id} ({username}), peers={list(self.peers.keys())}")
        self._ensure_processing()

    def remove_peer(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)
        with suppress(ValueError):
            self._peer_order.remove(peer_id)
        log.info(f"remove_peer {peer_id}, remaining={list(self.peers.keys())}")
        if not self.peers:
            self._stop_processing()

    def set_peer_track(self, peer_id: str, track: Any) -> None:
        peer = self.peers.get(peer_id)
        if peer is not None:
            peer.video_track = track
            log.info(f"set_peer_track {peer_id}")
        else:
            log.info(f"set_peer_track {peer_id} IGNORED — peer not registered")

    # ---- state & broadcasting -------------------------------------------

    def get_state_dict(self) -> dict:
        return {
            "type": "room_state",
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "username": p.username,
                    "has_video": p.video_track is not None,
                }
                for p in (
                    self.peers[pid]
                    for pid in self._peer_order
                    if pid in self.peers
                )
            ],
        }

    def _enqueue_to_all(self, msg: Any) -> None:
        for peer in self.peers.values():
            if peer.outgoing_queue is not None:
                with suppress(Exception):
                    peer.outgoing_queue.put_nowait(msg)

    async def broadcast_room_state(self) -> None:
        self._enqueue_to_all(self.get_state_dict())

    # ---- processing loop ------------------------------------------------

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

        import cv2
        import numpy as np
        from av import VideoFrame

        from core.compose import compose_frame

        out_pts = 0
        loop_iter = 0

        log.info("processing_loop STARTED")

        while not self._stopped:
            loop_iter += 1

            peers_with_tracks = [
                (pid, self.peers[pid])
                for pid in self._peer_order
                if pid in self.peers and self.peers[pid].video_track is not None
            ]
            n = len(peers_with_tracks)

            if n == 0:
                if loop_iter % 30 == 1:
                    log.info("processing_loop idle (no tracks)")
                black = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
                frame = VideoFrame.from_ndarray(black, format="bgr24")
                frame.pts = out_pts
                frame.time_base = Fraction(1, 30)
                out_pts += 1
                await self.broadcaster.publish(frame)
                await asyncio.sleep(0.1)
                continue

            raw_frames = await asyncio.gather(
                *(
                    asyncio.wait_for(peer.video_track.recv(), timeout=0.5)
                    for _, peer in peers_with_tracks
                ),
                return_exceptions=True,
            )

            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            cell_w = GRID_W // cols
            cell_h = GRID_H // rows
            canvas = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

            t_start = _time.perf_counter()
            model_ms_total = 0.0
            cells_drawn = 0
            use_face_swap = (
                self.face_swapper is not None
                and self.face_swapper.source_face is not None
            )

            for i, (pid, peer) in enumerate(peers_with_tracks):
                if isinstance(raw_frames[i], BaseException):
                    continue

                img = raw_frames[i].to_ndarray(format="bgr24")

                try:
                    s = _time.perf_counter()
                    if use_face_swap:
                        annotated = self.face_swapper(img)
                    else:
                        yolo_out = self._yolo(img)
                        annotated = compose_frame(
                            img, layer="detection", yolo=yolo_out,
                        )
                    model_ms_total += (_time.perf_counter() - s) * 1000.0
                except Exception:
                    annotated = img

                cell = cv2.resize(annotated, (cell_w, cell_h))
                cv2.putText(
                    cell,
                    peer.username,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                r, c = divmod(i, cols)
                y0, x0 = r * cell_h, c * cell_w
                canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
                cells_drawn += 1

            total_ms = (_time.perf_counter() - t_start) * 1000.0

            new_frame = VideoFrame.from_ndarray(canvas, format="bgr24")
            new_frame.pts = out_pts
            new_frame.time_base = Fraction(1, 30)
            out_pts += 1
            await self.broadcaster.publish(new_frame)

            mode_label = "face_swap" if use_face_swap else "yolo"
            self._frame_count += 1
            if self._frame_count % 30 == 1:
                log.info(
                    f"grid frame #{self._frame_count}: "
                    f"{cells_drawn}/{n} cells, "
                    f"mode={mode_label}, model={model_ms_total:.0f}ms, total={total_ms:.0f}ms, "
                    f"subs={len(self.broadcaster._subscribers)}"
                )
            if self._frame_count % self._timing_every_n == 0:
                self._enqueue_to_all(
                    {
                        "type": "timing",
                        "layer": mode_label,
                        "yolo_ms": model_ms_total if not use_face_swap else None,
                        "depth_ms": None,
                        "seg_ms": None,
                        "total_ms": total_ms,
                    }
                )
