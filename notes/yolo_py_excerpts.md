# Step 0 notes — reading fal-demos source before writing `app.py`

Sources read (all from `fal-ai-community/fal-demos@main`):
- `fal_demos/video/yolo_webcam_webrtc/yolo.py` ✓ (full raw source)
- `fal_demos/video/yolo_webcam_webrtc/yolo_client.py` ✓ (full raw source)
- `fal_demos/video/yolo_webcam_webrtc/frontend/src/main.js` ✓ (summarized)
- `fal_demos/video/matrix_webrtc/matrix.py` ✓ (summarized; discriminated-union control pattern)
- `fal_demos/image/sana.py` ✓ (class-kwargs + `local_python_modules`)

## The spec's single biggest misconception

> The spec assumes the `@fal.realtime` handler receives frames and returns frames in a message loop.

**This is wrong.** In `yolo.py`:

- `@fal.realtime("/realtime", buffering=5)` handles **WebRTC signaling only** — SDP offers/answers and ICE candidates, all Pydantic-discriminated.
- Video frames flow peer-to-peer via `aiortc` over the real WebRTC media transport. They never touch the realtime handler.
- Inference happens inside a custom `MediaStreamTrack.recv()` subclass (`YOLOTrack`). The peer connection's `@pc.on("track")` callback attaches the custom track to the outbound side: `pc.addTrack(create_yolo_track(incoming_track, self.yolo_model))`.
- Frame format at the boundary: `frame.to_ndarray(format="bgr24")` in, `VideoFrame.from_ndarray(annotated, format="bgr24")` out, preserving `pts` and `time_base`.

**Consequence for our build:** `core/frames.py` and the "msgpack binary JPEG vs base64 JSON" debate in §4.2/§6.1 of the spec are moot. Frames are RTP media and never get serialized by us. The relevant function is a `MultiPerceptionTrack(MediaStreamTrack)` with `async def recv(self)` calling the three models, not a message-loop handler.

## Answers to §5 (the 8 unknowns)

1. **Decorator & signature.**
   ```python
   @fal.realtime("/realtime", buffering=5)
   async def webrtc(
       self, inputs: AsyncIterator[RealtimeInput]
   ) -> AsyncIterator[RealtimeOutput]:
   ```
   It's an async generator. `inputs` yields Pydantic-validated signaling messages. We yield `RealtimeOutput` back.

2. **Frame ingress format.** Not relevant to the realtime handler. Inside `MediaStreamTrack.recv()`: `frame = await self.source_track.recv()` (an `av.VideoFrame`) → `frame.to_ndarray(format="bgr24")` gives a numpy `uint8` HxWx3 BGR array.

3. **Frame egress format.** Same: `VideoFrame.from_ndarray(arr, format="bgr24")`, then copy `pts` and `time_base` from the input frame.

4. **Per-connection state.** Lives on the `MediaStreamTrack` instance (`self.active_layer`, `self.latest_timings`, etc.). `yolo.py` stores `self.model` this way; we'll add mutable toggle state.

5. **Control messages.** `yolo.py`'s frontend uses **no data channel and no mid-session control messages** at all (confirmed: no `pc.createDataChannel` anywhere). `matrix.py` extends the same realtime endpoint with **additional discriminated-union input types** (`ActionInput`, `PauseInput`) sent over the same WebSocket that carries signaling. **This is the pattern we mirror.** Add `LayerToggleInput` to `RealtimeInputMessage`'s discriminated union; in `input_loop`, dispatch to a handler that mutates the track instance's `active_layer` field.

6. **`local_python_modules`.** Exact attribute name confirmed from `sana.py`: `local_python_modules = ["fal_demos"]` — a class-level attribute, a list of module names. We'll use `local_python_modules = ["core"]`.

7. **Class kwargs vs class attributes.** `sana.py` uses class kwargs for lifecycle config:
   ```python
   class Sana(
       fal.App,
       keep_alive=600,
       min_concurrency=0,
       max_concurrency=10,
       name="sana",
   ):
       local_python_modules = ["fal_demos"]
       machine_type = "GPU-H100"
       requirements = [...]
   ```
   `keep_alive`/`min_concurrency`/`max_concurrency`/`name` are **kwargs in the class definition**, not class-body attributes. `machine_type`, `requirements`, `local_python_modules` are class-body attributes. The spec's §6.1 has this wrong — fix it.

8. **TURN server.** ⚠️ **Required, not bundled.** `yolo.py`'s `setup()` reads `METERED_TURN_SECRET_KEY` and `METERED_TURN_LABEL` from env and **hard-fails at startup** if either is missing. `_build_ice_servers()` mints short-lived TURN credentials against `https://{label}.metered.live/api/v1/turn/credentials`. **This is a hard prerequisite for the build** — without a Metered account and those two secrets configured via `fal secrets`, even the unmodified tutorial won't start.

## §11 open-question answers

1. ✅ `@fal.realtime("/realtime", buffering=5)` (not `@fal.endpoint`).
2. ✅ Frame ingress is an `av.VideoFrame` from `aiortc`, at the `MediaStreamTrack.recv()` boundary — not inside the realtime handler at all.
3. ✅ Egress mirrors ingress: `VideoFrame.from_ndarray(arr, format="bgr24")` with copied `pts`/`time_base`.
4. ✅ Per-connection state = instance attributes on the custom `MediaStreamTrack`. Closures fine for things shared between the signaling loop and the track.
5. ✅ Controls go over the **signaling WebSocket**, extending `RealtimeInputMessage`'s discriminated union — the `matrix.py` pattern. No data channel needed.
6. ✅ `local_python_modules` — literal attribute name.
7. ✅ `fal run` prints the app ID; the client constructs `wss://fal.run/{owner}/{app}?fal_jwt_token=...` (from `main.js`) or uses `fal_client.AsyncClient().realtime(app_id, path="/realtime")` (from `yolo_client.py`). Both work against `fal run` and `fal deploy` identically.
8. ⚠️ See #8 above — **Metered TURN required**.
9. ✅ URL pattern: signaling at `wss://fal.run/<owner>/<app>/realtime` (or the JS form `wss://fal.run/<owner>/<app>` with a path-less connect — main.js normalizes). Normalizer helper is in `yolo_client.py::normalize_app_id`.
10. ⏳ Fal A100/H100 per-second pricing — not in source; look up on the fal pricing page before writing README §4.

## Other notable details from the source

- **YOLO weights path:** `YOLO("/data/yolov8n.pt")` — loaded from fal's `/data` persistent volume. The spec's worry about first-run `ultralytics` auto-download is sidestepped because the canonical path is pre-populated. For our build: either keep `/data/yolov8n.pt` and pre-upload weights, or let ultralytics auto-download on first `setup()` call and take the one-time cold-start tax. Document the choice.
- **Machine type in the existing example:** `GPU-H100`, not A100. The spec prescribes A100 for cost reasons; we can still override to A100, but we should note we're **downgrading from the upstream default** and be ready to justify it (it is justifiable: 3 small models, 80ms budget, ~900MB VRAM).
- **Requirements list (upstream):** `["aiortc", "av", "numpy", "opencv-python", "pydantic", "ultralytics"]` — no torch pin at all, no transformers. Torch comes transitively via `ultralytics`. We need to add `torch` (pinned), `transformers`, and a CUDA-matched extra-index-url for our two HF pipelines. Match `sana.py`'s style: `torch==2.6.0` + `--extra-index-url https://download.pytorch.org/whl/cu124`.
- **Buffering:** `@fal.realtime("/realtime", buffering=5)` — the `buffering=5` kwarg is undocumented in the spec but present in both `yolo.py` and `matrix.py`. Preserve it verbatim.
- **Shutdown hygiene:** `yolo.py` uses `MediaBlackhole` for inbound audio, `asyncio.Event` for stop, a `finally` block that closes PC/track in order. Mirror this exactly.
- **No tests, no `__init__.py` inside the demo dir.** It's a flat file. `sana.py` is the same. Matches the spec's ~150-line target.

## Pattern catalog (what to mirror where)

| Spec concern | Mirror from | How |
|---|---|---|
| Class definition, lifecycle kwargs | `sana.py` | `class MultiPerceptionWebRTC(fal.App, keep_alive=300, min_concurrency=0, max_concurrency=4, name="multi-perception-webrtc")` |
| `setup()` shape, multi-model loads | `sana.py` (`self.pipes` dict) | `self.yolo`, `self.depth`, `self.seg` as instance attrs; dummy-frame warmup per model |
| Signaling handler body | `yolo.py` | Copy `webrtc()` verbatim; only change is adding `LayerToggleInput` to the union and dispatching it |
| Control-message extension | `matrix.py` | Adding input types alongside `OfferInput` / `IceCandidateInput` in the discriminated union |
| Custom media track with inference | `yolo.py::create_yolo_track` | Expand `YOLOTrack.recv()` to run all 3 models; replace `draw_boxes` with our compose function; add `self.active_layer` |
| Client reference for testing | `yolo_client.py` | Use the Python client first for e2e sanity before touching the JS frontend |

## Build-order implications (revisions to §8)

- **Hour 0–1:** Clone fal-demos, copy `yolo_webcam_webrtc` to this repo, **get it running unmodified** against our fal account with Metered secrets configured. If Metered setup isn't available, the build cannot start — escalate immediately.
- **Hour 2:** Build `core/perception.py` against sample frames locally (CPU OK for validation). Unchanged from spec.
- **Hour 3:** Replace `YOLOTrack` with `MultiPerceptionTrack` that calls only YOLO first (single-model path working end-to-end).
- **Hour 4:** Add Depth + SegFormer to `setup()` and `MultiPerceptionTrack.recv()`. Sequential, timed.
- **Hour 5:** Add `LayerToggleInput` + handler + toggle state on the track. Compose function.
- **Hour 6:** Frontend — modify `main.js` (not `App.tsx`; spec had this wrong too) to add toggle buttons, a data-message sender, and HUD. Use `pc.getStats()` for FPS like the upstream does.
- **Hour 7–8:** unchanged.

## Still unknown / to verify on the fal account

- Whether our fal account has A100 availability (spec prescribes A100; upstream uses H100).
- Whether `fal secrets set METERED_TURN_SECRET_KEY ...` is available / already configured, or whether we need to create a Metered account first.
- The actual price per GPU-second on fal Serverless for the README cost math.
- Whether `fal run` for a realtime app on our account exposes a URL the local Vite frontend can reach directly.
