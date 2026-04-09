"""Multi-perception WebRTC on fal Serverless.

Three perceptual models (YOLOv8n, Depth Anything V2 Small, SegFormer-b0)
co-located on a single warm runner, driving a WebRTC webcam stream.

This file is a minimal, surgical extension of the upstream
``fal_demos/video/yolo_webcam_webrtc/yolo.py`` example:

* Same ``@fal.realtime("/realtime", buffering=5)`` signaling handler shape.
* Same Metered-TURN ICE bootstrap.
* Same ``aiortc`` peer-connection lifecycle and shutdown hygiene.

The three extensions, each small and isolated:

1. ``setup()`` loads three models and runs a black-frame warmup on each so
   the first user-visible frame is not paying cold-compile tax.
2. ``LayerToggleInput`` is added to the discriminated union of realtime
   inputs, and ``TimingOutput`` is added to the outputs. The existing
   signaling WebSocket carries both, mirroring the control-extension
   pattern already used by ``fal_demos/video/matrix_webrtc/matrix.py``.
3. The custom ``MultiPerceptionTrack`` replaces upstream's ``YOLOTrack``.
   It runs the models sequentially, lazily by layer, times each call,
   and pushes throttled ``TimingOutput`` messages into the signaling
   handler's outgoing queue for the frontend HUD.

Co-location, not parallelism, is the whole point: the three models are
CUDA-bound on the same GPU and would serialize on the same stream under
``asyncio.gather`` regardless — see ``notes/decisions.md``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

import fal

log = logging.getLogger("app")
log.setLevel(logging.DEBUG)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    log.addHandler(_h)
from fastapi import WebSocketDisconnect
from pydantic import BaseModel, Field, RootModel, TypeAdapter, ValidationError


# ---------- Pydantic signaling models ------------------------------------

class IceCandidate(BaseModel):
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class OfferInput(BaseModel):
    type: Literal["offer"]
    sdp: str


class IceCandidateInput(BaseModel):
    type: Literal["icecandidate"]
    candidate: IceCandidate | None = None


class LayerToggleInput(BaseModel):
    """Switch which model layer the runner composes into the outgoing frame."""

    type: Literal["layer"]
    layer: Literal["detection", "depth", "segmentation", "composite"]


class JoinInput(BaseModel):
    """Client requests to join the multiplayer room with a username."""

    type: Literal["join"]
    username: str


RealtimeInputMessage = Annotated[
    OfferInput | IceCandidateInput | LayerToggleInput | JoinInput,
    Field(discriminator="type"),
]


class RealtimeInput(RootModel):
    root: RealtimeInputMessage


class IceServersOutput(BaseModel):
    type: Literal["iceservers"]
    iceservers: list[dict]


class AnswerOutput(BaseModel):
    type: Literal["answer"]
    sdp: str


class IceCandidateOutput(BaseModel):
    type: Literal["icecandidate"]
    candidate: IceCandidate | None = None


class RunnerInfoOutput(BaseModel):
    """One-shot message sent right after ICE servers so the HUD can show
    warm-runner identity and which models are loaded."""

    type: Literal["runner_info"]
    runner_id: str
    models: list[str]


class TimingOutput(BaseModel):
    """Per-frame perception timings for the HUD. Throttled server-side to
    roughly 3 Hz so the WebSocket is not flooded."""

    type: Literal["timing"]
    layer: str
    yolo_ms: float | None = None
    depth_ms: float | None = None
    seg_ms: float | None = None
    total_ms: float


class ErrorOutput(BaseModel):
    type: Literal["error"]
    error: str


class PeerInfo(BaseModel):
    peer_id: str
    username: str
    has_video: bool


class JoinedOutput(BaseModel):
    """Sent to the joining client with their server-assigned peer ID."""

    type: Literal["joined"]
    peer_id: str


class RoomStateOutput(BaseModel):
    """Broadcast to all clients whenever the room changes."""

    type: Literal["room_state"]
    peers: list[PeerInfo]


RealtimeOutputMessage = Annotated[
    IceServersOutput
    | AnswerOutput
    | IceCandidateOutput
    | RunnerInfoOutput
    | TimingOutput
    | ErrorOutput
    | JoinedOutput
    | RoomStateOutput,
    Field(discriminator="type"),
]


class RealtimeOutput(RootModel):
    root: RealtimeOutputMessage


# ---------- The fal.App --------------------------------------------------

class MultiPerceptionWebRTC(
    fal.App,
    keep_alive=300,
    min_concurrency=0,
    max_concurrency=4,
    name="multi-perception-webrtc",
):
    # Mirrors upstream yolo_webcam_webrtc/yolo.py machine choice. Downgrading
    # to GPU-A100 is defensible (see README + notes/decisions.md) but we
    # match upstream here for zero-deviation on the canonical axis.
    machine_type = "GPU-H100"
    TURN_EXPIRY_SECONDS = 600

    local_python_modules = ["core"]

    requirements = [
        # Transport + WebRTC stack — match upstream yolo.py exactly.
        "aiortc",
        "av",
        "opencv-python",
        "pydantic",
        "ultralytics",
        # Perception stack for Depth Anything V2 + SegFormer.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.51.3",
        "pillow",
        "numpy<2",  # ultralytics + torch 2.6 still prefer numpy 1.x
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    # ------------------------------------------------------------------
    # setup() — loads models, warms each, creates the shared Room.
    # ------------------------------------------------------------------

    def setup(self):
        import os
        import uuid

        import numpy as np

        from core import detect_device
        from core.perception import (
            DepthEstimator,
            SemanticSegmenter,
            YoloDetector,
        )
        from core.room import Room

        self._metered_secret_key = os.getenv("METERED_TURN_SECRET_KEY")
        self._metered_label = os.getenv("METERED_TURN_LABEL")
        required = {
            "METERED_TURN_SECRET_KEY": self._metered_secret_key,
            "METERED_TURN_LABEL": self._metered_label,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing required Metered TURN env vars: {', '.join(missing)}. "
                "Set them via `fal secrets set` or .env before starting the realtime app."
            )

        fal_path = "/data/yolov8n.pt"
        yolo_weights = os.getenv(
            "YOLO_MODEL_PATH",
            fal_path if os.path.exists(fal_path) else "yolov8n.pt",
        )

        device = detect_device()
        print(f"setup: device={device}, yolo_weights={yolo_weights}")

        yolo = YoloDetector(weights_path=yolo_weights, device=device)
        depth_model = DepthEstimator(device=device)
        seg_model = SemanticSegmenter(device=device)

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = yolo(dummy)
        _ = depth_model(dummy)
        _ = seg_model(dummy)

        self.room = Room(yolo=yolo, depth=depth_model, seg=seg_model)

        self._runner_id = os.environ.get("FAL_RUNNER_ID") or str(uuid.uuid4())[:8]
        self._loaded_models = ["yolov8n", "depth-anything-v2-small", "segformer-b0"]

    # ------------------------------------------------------------------
    # Metered TURN bootstrap — copied verbatim from upstream yolo.py.
    # ------------------------------------------------------------------

    def _build_ice_servers(self) -> list[dict]:
        import json
        import urllib.parse
        import urllib.request

        label = self._metered_label
        secret_key = self._metered_secret_key
        assert label is not None
        assert secret_key is not None
        credentials_url = f"https://{label}.metered.live/api/v1/turn/credentials"
        credential_url = f"https://{label}.metered.live/api/v1/turn/credential"

        def fetch_ice_servers(api_key: str) -> list[dict]:
            query = urllib.parse.urlencode({"apiKey": api_key})
            join_char = "&" if "?" in credentials_url else "?"
            url = f"{credentials_url}{join_char}{query}"
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = response.read().decode("utf-8")
            raw_servers = json.loads(payload)
            servers: list[dict] = []
            for item in raw_servers:
                urls = item.get("urls")
                if not urls:
                    continue
                servers.append(
                    {
                        "urls": urls,
                        "username": item.get("username"),
                        "credential": item.get("credential", item.get("password")),
                    }
                )
            return servers

        query = urllib.parse.urlencode({"secretKey": secret_key})
        join_char = "&" if "?" in credential_url else "?"
        url = f"{credential_url}{join_char}{query}"
        body = json.dumps(
            {
                "expiryInSeconds": self.TURN_EXPIRY_SECONDS,
                "label": "fal-multi-perception-webrtc-demo",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8")
        credential_payload = json.loads(payload)
        temporary_api_key = credential_payload.get("apiKey")
        if not temporary_api_key:
            raise RuntimeError("Metered credential response missing apiKey.")

        servers = fetch_ice_servers(temporary_api_key)
        if not servers:
            raise RuntimeError("Metered returned empty ICE server list.")
        print("WebRTC: using Metered secret-minted ICE servers")
        return servers

    # ------------------------------------------------------------------
    # The realtime signaling endpoint — one invocation per WebSocket.
    # All invocations share self.room for multiplayer state.
    # ------------------------------------------------------------------

    @fal.realtime("/realtime", buffering=5)
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        import uuid

        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.sdp import candidate_from_sdp

        from core.room import create_broadcast_track

        peer_id = str(uuid.uuid4())[:8]
        peer_registered = False
        subscriber_ref: dict[str, object] = {"sub": None}
        incoming_video_ref: dict[str, object] = {"track": None}

        signal_ice_servers = self._build_ice_servers()
        rtc_ice_servers = [
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in signal_ice_servers
        ]
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=rtc_ice_servers)
        )
        blackhole = MediaBlackhole()
        stop_event = asyncio.Event()
        outgoing: asyncio.Queue = asyncio.Queue()
        input_adapter = TypeAdapter(RealtimeInputMessage)

        async def send(payload: RealtimeOutputMessage) -> None:
            if stop_event.is_set():
                return
            await outgoing.put(RealtimeOutput(root=payload))

        async def send_error(prefix: str, exc: Exception) -> None:
            await send(
                ErrorOutput(type="error", error=f"{prefix}:{type(exc).__name__}:{exc}")
            )
            stop_event.set()
            await outgoing.put(None)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                await send(IceCandidateOutput(type="icecandidate", candidate=None))
                return
            await send(
                IceCandidateOutput(
                    type="icecandidate",
                    candidate=IceCandidate(
                        candidate=candidate.candidate,
                        sdpMid=candidate.sdpMid,
                        sdpMLineIndex=candidate.sdpMLineIndex,
                    ),
                )
            )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()
                await outgoing.put(None)

        @pc.on("track")
        def on_track(track):
            log.info(f"app: on_track peer={peer_id}, kind={track.kind}, registered={peer_id in self.room.peers}")
            if track.kind == "video":
                incoming_video_ref["track"] = track
                if peer_id in self.room.peers:
                    self.room.set_peer_track(peer_id, track)
                sub = self.room.broadcaster.subscribe()
                subscriber_ref["sub"] = sub
                pc.addTrack(create_broadcast_track(sub))
                log.info(f"app: on_track peer={peer_id} — subscribed to broadcaster, broadcast_track added to PC")
            else:
                asyncio.ensure_future(blackhole.consume(track))

        async def handle_offer(payload: OfferInput) -> bool:
            try:
                offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await send(AnswerOutput(type="answer", sdp=pc.localDescription.sdp))
                return True
            except Exception as exc:
                await send_error("offer_failed", exc)
                return False

        async def handle_icecandidate(payload: IceCandidateInput) -> bool:
            try:
                candidate = payload.candidate
                if candidate is None:
                    await pc.addIceCandidate(None)
                    return True
                parsed = candidate_from_sdp(candidate.candidate)
                parsed.sdpMid = candidate.sdpMid
                parsed.sdpMLineIndex = candidate.sdpMLineIndex
                await pc.addIceCandidate(parsed)
                return True
            except Exception as exc:
                await send_error("ice_failed", exc)
                return False

        async def handle_join(payload: JoinInput) -> bool:
            nonlocal peer_registered
            log.info(f"app: handle_join peer={peer_id}, username={payload.username}, has_track={incoming_video_ref['track'] is not None}")
            self.room.add_peer(peer_id, payload.username, outgoing)
            peer_registered = True
            if incoming_video_ref["track"] is not None:
                self.room.set_peer_track(peer_id, incoming_video_ref["track"])
            await send(JoinedOutput(type="joined", peer_id=peer_id))
            await self.room.broadcast_room_state()
            return True

        async def handle_layer(payload: LayerToggleInput) -> bool:
            self.room.active_layer = payload.layer
            return True

        async def handle_payload(payload: RealtimeInputMessage) -> bool:
            if isinstance(payload, OfferInput):
                return await handle_offer(payload)
            if isinstance(payload, IceCandidateInput):
                return await handle_icecandidate(payload)
            if isinstance(payload, JoinInput):
                return await handle_join(payload)
            if isinstance(payload, LayerToggleInput):
                return await handle_layer(payload)
            return True

        async def input_loop() -> None:
            try:
                async for payload in inputs:
                    if stop_event.is_set():
                        break
                    try:
                        parsed_payload = (
                            payload.root
                            if isinstance(payload, RealtimeInput)
                            else input_adapter.validate_python(payload)
                        )
                    except ValidationError as exc:
                        await send_error("invalid_payload", exc)
                        break
                    should_continue = await handle_payload(parsed_payload)
                    if not should_continue:
                        break
            finally:
                stop_event.set()
                await outgoing.put(None)

        def _wrap_room_dict(msg: dict) -> RealtimeOutput | None:
            """Convert a plain dict from Room into a yield-ready RealtimeOutput."""
            msg_type = msg.get("type")
            if msg_type == "room_state":
                peers = [PeerInfo(**p) for p in msg.get("peers", [])]
                return RealtimeOutput(
                    root=RoomStateOutput(
                        type="room_state",
                        peers=peers,
                    )
                )
            if msg_type == "timing":
                return RealtimeOutput(root=TimingOutput(**msg))
            return None

        input_task: asyncio.Task | None = None
        try:
            await outgoing.put(
                RealtimeOutput(
                    root=IceServersOutput(
                        type="iceservers", iceservers=signal_ice_servers
                    )
                )
            )
            await outgoing.put(
                RealtimeOutput(
                    root=RunnerInfoOutput(
                        type="runner_info",
                        runner_id=self._runner_id,
                        models=self._loaded_models,
                    )
                )
            )
            input_task = asyncio.create_task(input_loop())
            while True:
                payload = await outgoing.get()
                if payload is None:
                    raise WebSocketDisconnect()
                if isinstance(payload, dict):
                    wrapped = _wrap_room_dict(payload)
                    if wrapped is not None:
                        yield wrapped
                    continue
                yield payload
        finally:
            stop_event.set()
            if peer_registered:
                self.room.remove_peer(peer_id)
                await self.room.broadcast_room_state()
            if subscriber_ref["sub"] is not None:
                self.room.broadcaster.unsubscribe(subscriber_ref["sub"])
            if input_task is not None:
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task
            await blackhole.stop()
            await pc.close()


# ---------- Local launcher (mirrors upstream yolo.py pattern) -----------

if __name__ == "__main__":
    info = MultiPerceptionWebRTC.spawn()
    print(f"App ID: {info.application}")
    print(f"Realtime endpoint: {info.application}/realtime")
    info.wait()
