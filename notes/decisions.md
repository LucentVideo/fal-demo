# Design decisions log

A rolling record of why each non-obvious choice in this repo was made the
way it was, with line references where the rationale is easy to lose.

## 1. Sequential inference, not `asyncio.gather`

`app.py::MultiPerceptionTrack.recv()` runs the three models one after the
other. It does **not** wrap them in `asyncio.gather`, `asyncio.wait`, or
any variant of "concurrent" scheduling.

Reason: all three models are CUDA-bound on the **same physical GPU**. A
single CUDA context schedules ops on the default stream; two coroutines
issuing kernels concurrently to the same stream do not get parallelism,
they get interleaved overhead. You would pay the `asyncio.gather` wrapper
cost and still serialize. Worse: the code would *look* parallel, and any
fal engineer reading it would flag the dishonesty immediately.

Sequential is honest. The total budget (~60–100 ms on an H100 for three
small models at 640×480) is fine for 10–15 FPS, which is the right
framerate band for the demo.

## 2. Lazy per-layer inference

Within `recv()`, single-layer modes only run the one model they need;
only composite runs all three. This gives single-layer modes framerate
headroom and keeps composite as the showpiece timing readout. The HUD
shows `—` for models that weren't called for the current frame — this
is more honest than running all three always to pad the numbers.

## 3. GPU-H100, matching upstream

`machine_type = "GPU-H100"` matches the upstream
`yolo_webcam_webrtc/yolo.py` exactly. The earlier version of the spec
prescribed A100 on cost grounds, which is defensible (900 MB of VRAM is
2.3 % of A100's 40 GB, and the 80 ms budget fits), but zero-deviation on
the canonical axis strengthens the "ready to merge into fal-demos"
framing, which is the primary pitch.

A/B note: if a future version targets cost-optimization messaging, the
swap is a one-line change plus an updated README §4.

## 4. Warmup in `setup()` with a black frame

Three calls to `self.<model>(np.zeros((480, 640, 3), np.uint8))` at the
end of `setup()`. This JIT-compiles CUDA kernels so the first
user-visible frame isn't the slow one. Mirrors the warmup idiom in
`fal_demos/image/sana.py::setup()`.

## 5. Toggle protocol: extend the signaling discriminated union

The upstream `yolo.py` frontend uses the signaling WebSocket for SDP
offer/answer and ICE candidates only — no data channel, no mid-session
control messages. `matrix.py` shows the canonical way to extend: add a
new Pydantic type (there: `ActionInput`, `PauseInput`) to the
discriminated union, dispatch it in the same `input_loop`, mutate server
state. We do exactly that with `LayerToggleInput`.

Ruled out: opening a separate `RTCDataChannel`. It would double the
plumbing (client + server, open/close lifecycle, error handling) for a
three-type message we can ship in four lines of Pydantic on an already-
open WebSocket.

## 6. `TimingOutput` for HUD, throttled to ~3 Hz

The track's `recv()` pushes a `TimingOutput` into the signaling handler's
outgoing queue every 5 frames (so ~3 Hz at 15 FPS). Ruled out: overlaying
the timings onto the frame with `cv2.putText` — it's simpler but less
readable and couples the HUD to the media plane. Ruled out: every-frame
push — the WebSocket can take it but the DOM update rate is wasted work.

Cross-loop safety: the track runs inside aiortc on the same asyncio loop
as the signaling handler (fal runs one loop per handler), so
`outgoing.put_nowait(...)` is safe. Wrapped in `suppress(Exception)` to
fail silently if the queue state ever gets weird.

## 7. `local_python_modules = ["core"]`

Packages the three wrapper files and `compose.py` onto the runner.
Mirrors `fal_demos/image/sana.py`'s `local_python_modules = ["fal_demos"]`.
Ruled out: inlining all perception code into `app.py` — would push the
single file past the ~150-line spec target and mix pure-python perception
logic with the WebRTC handler.

## 8. `keep_alive=300`, `min_concurrency=0`, `max_concurrency=4`

`keep_alive=300` is longer than the canvas demos (60 s) because an
interactive demo session runs 5+ minutes and we don't want cold bounces
mid-session; shorter than a production setting (900+) because idle cost
matters.

`min_concurrency=0` respects the Serverless scale-to-zero pricing model.
One concession: for the live cofounder demo, consider temporarily setting
it to 1 so there's always a warm runner during the meeting. Toggle back
after.

## 9. Depth normalization per-frame

`DepthEstimator.__call__` normalizes to the frame's own min/max before
colormapping. This makes the depth layer readable at all times at the
cost of losing absolute-depth comparability between frames. For a demo
the trade is obviously correct; for a robotics client consuming
absolute depth, expose raw metric depth alongside.

## 10. Frame ingress/egress via aiortc, not through the realtime handler

The spec's original `app.py` shape had a message loop receiving frames
from `connection`. That was wrong: `@fal.realtime` carries only signaling
(SDP / ICE / control) in the Pydantic-discriminated union; video frames
flow peer-to-peer over RTP via aiortc. The inference runs inside a
custom `MediaStreamTrack.recv()`, returning `av.VideoFrame` objects.
This is exactly how upstream `YOLOTrack` works; we just extend the same
shape to three models.

One upshot: the "msgpack binary vs base64 JSON" debate in the original
spec is moot — we never serialize a frame ourselves. RTP handles it.
