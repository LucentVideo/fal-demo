"""Multi-perception WebRTC on lucent-serverless.

Port of ``app_fal.py`` to the lucent-serverless runtime. The realtime
handler body is byte-identical — lucent's ``@ls.realtime`` mirrors fal's
``AsyncIterator[PydanticModel]`` contract — so only the App class config
and imports differ.

Ported pieces:

* ``fal.App`` → ``ls.App`` with class-attribute config (``app_id``,
  ``compute_type``, ``machine_type`` using the RunPod GPU id, etc.).
* ``requirements`` trimmed: the lucent GPU base image already bakes
  torch/torchvision/opencv-python/pydantic/pillow/numpy/transformers/
  huggingface_hub/onnxruntime-gpu/cupy-cuda12x, so we only install the
  WebRTC-specific extras here.
* ``local_python_modules = ["core"]`` is not needed — lucent tars and
  uploads the entire source directory, so ``core/`` ships automatically.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

import lucent_serverless as ls

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
    layer: Literal["detection"]


class JoinInput(BaseModel):
    """Client requests to join the multiplayer room with a username."""

    type: Literal["join"]
    username: str


class SetSourceFaceInput(BaseModel):
    """Client uploads a source face image (base64 JPEG/PNG)."""

    type: Literal["set_source_face"]
    image_data: str


class ClearSourceFaceInput(BaseModel):
    """Client clears the current source face."""

    type: Literal["clear_source_face"]


class ToggleEnhanceInput(BaseModel):
    """Client toggles GFPGAN face enhancement."""

    type: Literal["toggle_enhance"]
    enabled: bool


class ShuffleFacesInput(BaseModel):
    """Client requests a random face shuffle among all captured peers."""

    type: Literal["shuffle_faces"]


RealtimeInputMessage = Annotated[
    OfferInput
    | IceCandidateInput
    | LayerToggleInput
    | JoinInput
    | SetSourceFaceInput
    | ClearSourceFaceInput
    | ToggleEnhanceInput
    | ShuffleFacesInput,
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
    total_ms: float


class ErrorOutput(BaseModel):
    type: Literal["error"]
    error: str


class PeerInfo(BaseModel):
    peer_id: str
    username: str
    has_video: bool
    face_captured: bool = False


class JoinedOutput(BaseModel):
    """Sent to the joining client with their server-assigned peer ID."""

    type: Literal["joined"]
    peer_id: str


class RoomStateOutput(BaseModel):
    """Broadcast to all clients whenever the room changes."""

    type: Literal["room_state"]
    peers: list[PeerInfo]
    face_override_active: bool = False
    face_override_by: str | None = None
    face_override_image: str | None = None


class SourceFaceSetOutput(BaseModel):
    """Sent to all clients when the source face changes."""

    type: Literal["source_face_set"]
    success: bool
    message: str = ""
    set_by: str | None = None
    image_data: str | None = None


class FaceCapturedOutput(BaseModel):
    """Broadcast when a peer's face is auto-detected from their webcam."""

    type: Literal["face_captured"]
    peer_id: str
    username: str
    success: bool


class ShuffleAssignment(BaseModel):
    peer_id: str
    username: str
    assigned_face_of: str


class ShuffleAppliedOutput(BaseModel):
    """Broadcast when a face shuffle is applied."""

    type: Literal["shuffle_applied"]
    assignments: list[ShuffleAssignment]


class ShuffleClearedOutput(BaseModel):
    """Broadcast when the face shuffle is cleared."""

    type: Literal["shuffle_cleared"]
    reason: str = ""


RealtimeOutputMessage = Annotated[
    IceServersOutput
    | AnswerOutput
    | IceCandidateOutput
    | RunnerInfoOutput
    | TimingOutput
    | ErrorOutput
    | JoinedOutput
    | RoomStateOutput
    | SourceFaceSetOutput
    | FaceCapturedOutput
    | ShuffleAppliedOutput
    | ShuffleClearedOutput,
    Field(discriminator="type"),
]


class RealtimeOutput(RootModel):
    root: RealtimeOutputMessage


# ---------- The lucent App -----------------------------------------------

class MultiPerceptionWebRTC(ls.App):
    app_id = "multi-perception-webrtc"
    compute_type = "GPU"
    # RunPod GPU id; the lucent GPU base image is built on runpod/pytorch
    # cu128 which pairs with H100s.
    machine_type = "H100 80GB"
    cloud_type = "SECURE"
    container_disk_gb = 40
    keep_alive = 300
    min_concurrency = 0
    max_concurrency = 4

    TURN_EXPIRY_SECONDS = 600

    # Only the extras the base GPU image doesn't already have.
    # Base already bakes: torch, torchvision, opencv-python, pydantic,
    # pillow, numpy<2, transformers, huggingface_hub, onnxruntime-gpu,
    # cupy-cuda12x (see lucent-serverless/images/gpu/Dockerfile).
    requirements = [
        "aiortc",
        "av",
        "ultralytics",
        "insightface",
        "gfpgan",
    ]

    # Only ship these — the repo root has frontend/, gfpgan/ weights,
    # upstream-fal-demos/, etc. that the pod doesn't need.
    include = ["app.py", "core"]

    # ------------------------------------------------------------------
    # setup() — loads models, warms each, creates the shared Room.
    # ------------------------------------------------------------------

    def setup(self):
        import os
        import uuid

        import numpy as np

        from core import detect_device
        from core.face_swap import FaceSwapper
        from core.yolo import YoloDetector
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
                "Set them via the controller env or .env before starting the realtime app."
            )

        # On lucent pods user code lands in /opt/app (GPU) or /app (CPU);
        # we keep the yolov8n.pt path flexible via env var.
        yolo_weights = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

        device = detect_device()
        print(f"setup: device={device}, yolo_weights={yolo_weights}")

        yolo = YoloDetector(weights_path=yolo_weights, device=device)

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = yolo(dummy)

        face_swapper = FaceSwapper(device=device)
        face_swapper.warmup()

        self.room = Room(yolo=yolo, face_swapper=face_swapper)

        self._runner_id = os.environ.get("LUCENT_APP_ID") or str(uuid.uuid4())[:8]
        self._loaded_models = [
            "yolov8n",
            "insightface-buffalo_l", "inswapper_128_fp16",
        ]
        # Minted TURN credentials last TURN_EXPIRY_SECONDS; refresh cache before then.
        self._metered_ice_cache: list[dict] | None = None
        self._metered_ice_expires: float = 0.0

    # ------------------------------------------------------------------
    # Metered TURN bootstrap — identical to app_fal.py.
    # ------------------------------------------------------------------

    def _fetch_metered_ice_servers_sync(self) -> list[dict]:
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
                "label": "lucent-multi-perception-webrtc-demo",
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
        print("WebRTC: minted fresh Metered ICE servers")
        return servers

    # ------------------------------------------------------------------
    # The realtime signaling endpoint — one invocation per WebSocket.
    # All invocations share self.room for multiplayer state.
    # ------------------------------------------------------------------

    @ls.realtime("/realtime", buffering=5)
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        import uuid

        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCRtpCodecCapability,
            RTCSessionDescription,
        )
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.sdp import candidate_from_sdp

        from core.room import create_broadcast_track

        import time

        peer_id = str(uuid.uuid4())[:8]
        peer_registered = False
        subscriber_ref: dict[str, object] = {"sub": None}
        incoming_video_ref: dict[str, object] = {"track": None}

        now = time.time()
        cache_ttl = min(480.0, float(self.TURN_EXPIRY_SECONDS) - 60.0)
        if (
            self._metered_ice_cache is not None
            and now < self._metered_ice_expires
        ):
            signal_ice_servers = self._metered_ice_cache
            print("WebRTC: using cached Metered ICE servers")
        else:
            signal_ice_servers = await asyncio.to_thread(
                self._fetch_metered_ice_servers_sync
            )
            self._metered_ice_cache = signal_ice_servers
            self._metered_ice_expires = now + cache_ttl
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
                broadcast_track = create_broadcast_track(sub)
                transceiver = pc.addTrack(broadcast_track)
                # Prefer H264 for higher quality ceiling (3 Mbps vs VP8's 1.5 Mbps)
                try:
                    h264_codec = RTCRtpCodecCapability(
                        mimeType="video/H264", clockRate=90000
                    )
                    transceiver.setCodecPreferences([h264_codec])
                except Exception as e:
                    log.warning(f"Could not set H264 preference: {e}")
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

        async def handle_set_source_face(payload: SetSourceFaceInput) -> bool:
            import base64
            import cv2
            import numpy as np

            try:
                raw = payload.image_data
                if "," in raw:
                    raw = raw.split(",", 1)[1]
                img_bytes = base64.b64decode(raw)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("could not decode image")
            except Exception as exc:
                log.info(f"app: set_source_face decode error: {exc}")
                self.room._enqueue_to_all({
                    "type": "source_face_set",
                    "success": False,
                    "message": f"Failed to decode image: {exc}",
                })
                return True

            success = self.room.face_swapper.set_source(img)
            peer = self.room.peers.get(peer_id)
            username = peer.username if peer else peer_id
            if success:
                self.room._face_override_by = username
                self.room._face_override_image = payload.image_data
            self.room._enqueue_to_all({
                "type": "source_face_set",
                "success": success,
                "message": "Source face set" if success else "No face detected in image",
                "set_by": username if success else None,
                "image_data": payload.image_data if success else None,
            })
            log.info(f"app: set_source_face success={success}, set_by={username}")
            return True

        async def handle_clear_source_face(_payload: ClearSourceFaceInput) -> bool:
            self.room.face_swapper.clear_source()
            self.room._face_override_by = None
            self.room._face_override_image = None
            self.room._enqueue_to_all({
                "type": "source_face_set",
                "success": True,
                "message": "Source face cleared",
                "set_by": None,
                "image_data": None,
            })
            return True

        async def handle_toggle_enhance(payload: ToggleEnhanceInput) -> bool:
            self.room.face_swapper.enhance_enabled = payload.enabled
            log.info(f"app: enhance_enabled={payload.enabled}")
            return True

        async def handle_shuffle_faces(_payload: ShuffleFacesInput) -> bool:
            self.room.shuffle()
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
            if isinstance(payload, SetSourceFaceInput):
                return await handle_set_source_face(payload)
            if isinstance(payload, ClearSourceFaceInput):
                return await handle_clear_source_face(payload)
            if isinstance(payload, ToggleEnhanceInput):
                return await handle_toggle_enhance(payload)
            if isinstance(payload, ShuffleFacesInput):
                return await handle_shuffle_faces(payload)
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
                        face_override_active=msg.get("face_override_active", False),
                        face_override_by=msg.get("face_override_by"),
                        face_override_image=msg.get("face_override_image"),
                    )
                )
            if msg_type == "timing":
                return RealtimeOutput(root=TimingOutput(**msg))
            if msg_type == "source_face_set":
                return RealtimeOutput(root=SourceFaceSetOutput(**msg))
            if msg_type == "face_captured":
                return RealtimeOutput(root=FaceCapturedOutput(**msg))
            if msg_type == "shuffle_applied":
                assignments = [ShuffleAssignment(**a) for a in msg.get("assignments", [])]
                return RealtimeOutput(
                    root=ShuffleAppliedOutput(type="shuffle_applied", assignments=assignments)
                )
            if msg_type == "shuffle_cleared":
                return RealtimeOutput(root=ShuffleClearedOutput(**msg))
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


# ---------- Local launcher ------------------------------------------------

if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    # Forward secrets the pod needs at setup() time.
    pod_env = {
        k: v
        for k, v in {
            "METERED_TURN_SECRET_KEY": os.environ.get("METERED_TURN_SECRET_KEY"),
            "METERED_TURN_LABEL": os.environ.get("METERED_TURN_LABEL"),
        }.items()
        if v
    }

    info = MultiPerceptionWebRTC.spawn(env=pod_env)
    print(f"App ID: {info.app_id}")
    print(f"Realtime endpoint: {info.pod_url}/realtime")
    print(f"WebSocket URL:     {info.pod_url.replace('https://', 'wss://')}/realtime")
