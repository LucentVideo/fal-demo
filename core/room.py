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
import functools
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
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


class LatestFrameRelay:
    """Continuously drains a WebRTC track, keeping only the most recent frame.

    Without this, ``track.recv()`` returns the *next* buffered frame (FIFO),
    so frames pile up when the processing loop is slower than the sender's
    frame-rate.  With N peers the backlog grows fastest for the earliest
    joiner — exactly the lag pattern reported.

    The relay runs a tight background task that calls ``track.recv()`` in a
    loop and atomically overwrites ``self.latest``.  The processing loop
    reads ``grab()`` which returns the newest frame instantly (or *None* if
    no frame has arrived yet).
    """

    def __init__(self, track: Any) -> None:
        self._track = track
        self.latest: Any = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._drain())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _drain(self) -> None:
        """Read frames as fast as the track produces them, keep only the last."""
        try:
            while True:
                frame = await self._track.recv()
                self.latest = frame          # atomic reference swap
        except (asyncio.CancelledError, Exception):
            return

    def grab(self) -> Any:
        """Return the most recent frame, or None if nothing arrived yet."""
        return self.latest


@dataclass
class PeerState:
    """Mutable state for a single connected peer."""

    peer_id: str
    username: str
    video_track: Any = None
    frame_relay: LatestFrameRelay | None = None
    outgoing_queue: asyncio.Queue | None = None
    reference_face: Any = None
    face_captured: bool = False


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
    import av
    from aiortc.mediastreams import MediaStreamTrack

    class BroadcastTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._sub = subscriber
            self._first = True

        async def recv(self):
            frame = await self._sub.recv()
            # Force the first frame each subscriber sees to be an IDR.
            # Without this, a late joiner's first packet is a P-frame
            # referencing a keyframe they never received, and the browser
            # hangs on "buffering first frame" until PLI recovery (which
            # often fails under cold-start latency).
            if self._first:
                try:
                    frame.pict_type = av.video.frame.PictureType.I
                except Exception:
                    pass
                self._first = False
            return frame

        def stop(self) -> None:
            super().stop()
            self._sub.close()

    return BroadcastTrack()


GRID_W = 1920
GRID_H = 1080


def _letterbox(img, target_w: int, target_h: int):
    """Resize *img* to fit inside target_w x target_h, preserving aspect ratio.

    The image is scaled uniformly and centered; remaining area is black.
    """
    import cv2

    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    import numpy as np
    cell = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_y = (target_h - new_h) // 2
    pad_x = (target_w - new_w) // 2
    cell[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return cell


def _draw_peer_username(cell, username: str, cell_h: int) -> None:
    """Username overlay on each grid cell; scales with cell height."""
    import cv2

    if not username:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Large, readable labels; scales with tile height (solo view gets the cap).
    font_scale = max(1.45, min(3.6, cell_h * 0.0075))
    thickness = max(3, min(7, int(round(font_scale * 1.6))))
    margin = max(12, int(cell_h * 0.045))
    (_, th), _ = cv2.getTextSize(username, font, font_scale, thickness)
    y = min(cell.shape[0] - 1, margin + th)
    cv2.putText(
        cell,
        username,
        (margin, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


class Room:
    """Single multiplayer room shared across all WebSocket handlers."""

    def __init__(self, *, yolo: Any, face_swapper: Any = None) -> None:
        self.peers: dict[str, PeerState] = {}
        self._peer_order: list[str] = []
        self.active_layer: str = "detection"
        self.broadcaster = FrameBroadcaster()

        self._yolo = yolo
        self.face_swapper = face_swapper

        self._processing_task: asyncio.Task | None = None
        self._stopped = False
        self._frame_count = 0
        self._timing_every_n = 5
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._shuffle_mapping: dict[str, Any] = {}
        self._shuffle_labels: dict[str, str] = {}

        self._face_override_by: str | None = None
        self._face_override_image: str | None = None

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
        peer = self.peers.pop(peer_id, None)
        if peer is not None:
            if peer.frame_relay is not None:
                peer.frame_relay.stop()
            peer.reference_face = None
        with suppress(ValueError):
            self._peer_order.remove(peer_id)
        if peer_id in self._shuffle_mapping and self._shuffle_mapping:
            self.clear_shuffle(reason="a player left the room")
        log.info(f"remove_peer {peer_id}, remaining={list(self.peers.keys())}")
        if not self.peers:
            self._stop_processing()

    def set_peer_track(self, peer_id: str, track: Any) -> None:
        peer = self.peers.get(peer_id)
        if peer is not None:
            # Stop any previous relay for this peer.
            if peer.frame_relay is not None:
                peer.frame_relay.stop()
            peer.video_track = track
            relay = LatestFrameRelay(track)
            relay.start()
            peer.frame_relay = relay
            log.info(f"set_peer_track {peer_id} — relay started")
        else:
            log.info(f"set_peer_track {peer_id} IGNORED — peer not registered")

    # ---- state & broadcasting -------------------------------------------

    def get_state_dict(self) -> dict:
        d: dict[str, Any] = {
            "type": "room_state",
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "username": p.username,
                    "has_video": p.video_track is not None,
                    "face_captured": p.face_captured,
                }
                for p in (
                    self.peers[pid]
                    for pid in self._peer_order
                    if pid in self.peers
                )
            ],
            "face_override_active": self._face_override_by is not None,
            "face_override_by": self._face_override_by,
        }
        if self._face_override_image is not None:
            d["face_override_image"] = self._face_override_image
        return d

    def _enqueue_to_all(self, msg: Any) -> None:
        for peer in self.peers.values():
            if peer.outgoing_queue is not None:
                with suppress(Exception):
                    peer.outgoing_queue.put_nowait(msg)

    async def broadcast_room_state(self) -> None:
        self._enqueue_to_all(self.get_state_dict())

    # ---- face capture ----------------------------------------------------

    def _try_capture_face(self, peer: PeerState, img_bgr) -> bool:
        """Attempt to detect and store a reference face for this peer.

        Returns True if a face was successfully captured.
        """
        if self.face_swapper is None:
            return False
        faces = self.face_swapper.detect_faces(img_bgr)
        if not faces:
            return False
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        peer.reference_face = best
        peer.face_captured = True
        log.info(f"face captured for {peer.peer_id} ({peer.username})")
        self._enqueue_to_all({
            "type": "face_captured",
            "peer_id": peer.peer_id,
            "username": peer.username,
            "success": True,
        })
        return True

    def get_face_map(self) -> dict[str, Any]:
        """Return {peer_id: reference_face} for all peers with captured faces."""
        return {
            pid: peer.reference_face
            for pid, peer in self.peers.items()
            if peer.face_captured and peer.reference_face is not None
        }

    # ---- shuffle ---------------------------------------------------------

    def shuffle(self) -> bool:
        """Generate a random derangement of captured faces across peers.

        Uses Sattolo's algorithm to guarantee no peer gets their own face.
        Returns True if shuffle was applied, False if not enough faces.
        """
        face_map = self.get_face_map()
        if len(face_map) < 2:
            self._enqueue_to_all({
                "type": "shuffle_cleared",
                "reason": "Need at least 2 captured faces to shuffle",
            })
            return False

        pids = list(face_map.keys())
        n = len(pids)
        shuffled = pids[:]
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i - 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        self._shuffle_mapping = {
            pids[k]: face_map[shuffled[k]] for k in range(n)
        }
        self._shuffle_labels = {
            pids[k]: self.peers[shuffled[k]].username for k in range(n)
        }

        assignments = [
            {
                "peer_id": pids[k],
                "username": self.peers[pids[k]].username,
                "assigned_face_of": self.peers[shuffled[k]].username,
            }
            for k in range(n)
        ]
        self._enqueue_to_all({
            "type": "shuffle_applied",
            "assignments": assignments,
        })
        log.info(f"shuffle applied: {self._shuffle_labels}")
        return True

    def clear_shuffle(self, reason: str = "shuffle cleared") -> None:
        self._shuffle_mapping.clear()
        self._shuffle_labels.clear()
        self._enqueue_to_all({
            "type": "shuffle_cleared",
            "reason": reason,
        })
        log.info(f"shuffle cleared: {reason}")

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
        self._executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(max_workers=2)

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
                if pid in self.peers and self.peers[pid].frame_relay is not None
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

            # Grab the latest frame from each peer's relay — instant, no
            # waiting, no buffer accumulation.  If a relay hasn't received
            # its first frame yet we get None and skip that peer below.
            raw_frames = [peer.frame_relay.grab() for _, peer in peers_with_tracks]

            # If every relay returned None (no frames arrived yet), yield
            # briefly instead of busy-spinning.
            if all(f is None for f in raw_frames):
                await asyncio.sleep(0.01)
                continue

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

            loop = asyncio.get_running_loop()

            # ----------------------------------------------------------
            # Build per-peer processing coroutines so we can fan them out
            # concurrently with asyncio.gather (Tier 1.5).  GPU inference
            # is still serialised by the GIL / CUDA stream, but CPU-side
            # work (ndarray conversion, resize, text overlay) overlaps.
            # ----------------------------------------------------------
            async def _process_peer(idx, pid, peer):
                if raw_frames[idx] is None:
                    return idx, pid, peer, None, 0.0

                img = raw_frames[idx].to_ndarray(format="bgr24")

                if not peer.face_captured and self.face_swapper is not None:
                    try:
                        await loop.run_in_executor(
                            self._executor,
                            self._try_capture_face, peer, img,
                        )
                    except Exception:
                        pass

                try:
                    s = _time.perf_counter()
                    if use_face_swap:
                        annotated = await loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                self.face_swapper.swap_with_source,
                                img, self.face_swapper.source_face,
                                copy=False,
                            ),
                        )
                    elif pid in self._shuffle_mapping:
                        annotated = await loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                self.face_swapper.swap_with_source,
                                img, self._shuffle_mapping[pid],
                                copy=False,
                            ),
                        )
                    else:
                        yolo_out = await loop.run_in_executor(
                            self._executor, self._yolo, img,
                        )
                        annotated = compose_frame(img, yolo=yolo_out)
                    elapsed = (_time.perf_counter() - s) * 1000.0
                except Exception:
                    annotated = img
                    elapsed = 0.0

                return idx, pid, peer, annotated, elapsed

            tasks = [
                _process_peer(i, pid, peer)
                for i, (pid, peer) in enumerate(peers_with_tracks)
            ]
            results = await asyncio.gather(*tasks)

            for idx, pid, peer, annotated, elapsed in results:
                if annotated is None:
                    continue
                model_ms_total += elapsed

                cell = _letterbox(annotated, cell_w, cell_h)
                _draw_peer_username(cell, peer.username, cell_h)

                r, c = divmod(idx, cols)
                y0, x0 = r * cell_h, c * cell_w
                canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
                cells_drawn += 1

            total_ms = (_time.perf_counter() - t_start) * 1000.0

            new_frame = VideoFrame.from_ndarray(canvas, format="bgr24")
            new_frame.pts = out_pts
            new_frame.time_base = Fraction(1, 30)
            out_pts += 1
            await self.broadcaster.publish(new_frame)

            if use_face_swap:
                mode_label = "face_swap"
            elif self._shuffle_mapping:
                mode_label = "shuffle"
            else:
                mode_label = "yolo"
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
                        "yolo_ms": model_ms_total if mode_label == "yolo" else None,
                        "total_ms": total_ms,
                    }
                )
