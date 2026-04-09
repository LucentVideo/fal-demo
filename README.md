# Multi-Perception WebRTC on fal Serverless

A custom `fal.App` that extends the
[`yolo_webcam_webrtc`](https://github.com/fal-ai-community/fal-demos/tree/main/fal_demos/video/yolo_webcam_webrtc)
example to run **three perceptual models — YOLOv8n, Depth Anything V2 Small,
and SegFormer-b0 — in parallel on every frame from a single warm runner**.
The browser opens a webcam stream over WebRTC and shows the same stream
processed by any combination of the three models, switchable live from a
toggle row above the processed panel. End-to-end latency is roughly
60–100 ms in single-model mode and 100–140 ms in composite mode on an H100.

## Why this shape is the right one for production realtime vision

The YOLO-on-webcam tutorial is the toy case. Real applications — AR try-on,
robotic teleop, creative filters, agentic video — need a _stack_ of
perceptual models running in lockstep on every frame, not one. The binding
constraint on a stream is **per-frame latency budget**, not throughput or
cost per call. Network round-trips between separate runners, or between a
runner and marketplace endpoints, blow that budget: a 30–60 ms RTT per hop
turns three models into a 300 ms frame. The only way to fit N models inside
~80 ms is to co-locate them in `setup()` on one runner. fal Serverless is
the only place that pattern is clean, and this file is the minimal
demonstration of it.

## What's here

```
app.py                 # the fal.App — a minimal extension of yolo.py
pyproject.toml         # uv project with all runtime deps
.env.example           # template for Metered TURN credentials
core/
  perception.py        # three model wrappers (YOLO / Depth / Seg), pure python
  compose.py           # four render modes (Detection / Depth / Seg / Composite)
frontend/
  package.json         # self-contained Vite project
  vite.config.js       # dev server config
  index.html           # UI with local-mode toggle + layer toggles + HUD
  src/main.js          # WebRTC signaling, layer toggles, HUD, timing handler
  src/style.css        # toggle-button + HUD styles
notes/
  yolo_py_excerpts.md  # everything learned by reading the upstream source
  decisions.md         # rolling record of non-obvious choices
```

## Design decisions, with line references to the canonical fal-demos patterns we mirrored

- **`setup()` loads all three models and warms each with a black frame.**
  Mirrors the load-then-warm idiom in `fal_demos/image/sana.py::setup()`.
  Ensures CUDA kernels are JIT-compiled before the first real request,
  keeping cold-compile tax off the user-visible latency budget.
  → `app.py:193-208`

- **The three models are called sequentially, not via `asyncio.gather`.**
  All three are CUDA-bound on the same GPU and would serialize on the same
  stream under gather regardless; sequential is honest, and the total
  budget is fine. → `app.py::MultiPerceptionTrack.recv` and
  `notes/decisions.md §1`.

- **Layer toggles extend the existing signaling WebSocket's discriminated
  union**, mirroring the extension pattern in
  `fal_demos/video/matrix_webrtc/matrix.py` (`ActionInput`, `PauseInput`).
  No separate `RTCDataChannel` needed. → `app.py::LayerToggleInput` and
  `app.py::handle_layer`.

- **`local_python_modules = ["core"]`** packages the local `core/`
  perception and compose modules onto the runner. Same pattern as
  `fal_demos/image/sana.py::local_python_modules = ["fal_demos"]`.

- **`machine_type = "GPU-H100"`** matches upstream
  `yolo_webcam_webrtc/yolo.py` exactly, zero-deviation on the canonical
  axis. A downgrade to `GPU-A100` is defensible (2.3 % of A100 VRAM, 80 ms
  budget fits) if the messaging needs to lean on cost; see
  `notes/decisions.md §3`.

- **`keep_alive=300`, `min_concurrency=0`, `max_concurrency=4`** as class
  kwargs on the `fal.App` subclass — same style as `sana.py`.
  `keep_alive=300` because realistic interactive sessions are 5+ minutes
  but 300 s caps idle cost.

- **Per-frame perception timings are pushed over the signaling WebSocket
  as `TimingOutput` messages**, throttled to roughly 3 Hz from inside the
  track's `recv()`. The frontend listens for `type: "timing"` in
  `ws.onmessage` and updates the HUD chips. → `app.py::TimingOutput` and
  `frontend/src/main.js::applyTimingMessage`.

- **Lazy per-layer inference**: single-layer modes only run the one model
  they need; only `composite` runs all three. The HUD shows `—` for models
  that weren't called for the current frame rather than padding timings.

## VRAM and cost math

| Model                   | Weights | Activations @ 640×480 | Total       |
| ----------------------- | ------- | --------------------- | ----------- |
| YOLOv8n                 | ~12 MB  | ~150 MB               | ~160 MB     |
| Depth Anything V2 Small | ~140 MB | ~300 MB               | ~440 MB     |
| SegFormer-b0            | ~14 MB  | ~250 MB               | ~260 MB     |
| **Total**               |         |                       | **~920 MB** |

~920 MB on an 80 GB H100 is ~1.2 %. This is exactly why co-location is the
right pattern — there is no resource conflict, just amortized loading and
zero network hops. A single warm runner is strictly cheaper than three
marketplace endpoints called per frame, because the per-frame cost of three
model calls over the internal network swamps the cost of keeping one runner
warm for a session.

> Fill in exact $/GPU-second on the day from fal's pricing page; the ratio
> holds regardless of the absolute numbers.

## What this isn't

- **Not a multi-GPU demo.** For multi-GPU, use `fal.distributed` and see
  `fal_demos/image/parallel_sdxl`. This demo is about co-locating several
  models on one GPU, which is a different pattern.
- **Not a generative demo.** An SDXL-Turbo img2img version of this is a
  different fork. The pitch here is the _perception stack_, not a
  prompt-driven transform.
- **Not optimized for max framerate.** TensorRT or ONNX conversions would
  shave another 30 % off the per-frame budget. That's a hardening pass,
  not a one-day tutorial extension.
- **Not production-hardened.** No rate limiting, no auth on the realtime
  endpoint beyond fal's defaults, no observability beyond `print`s. A real
  deployment would add all three.

## How this could merge into fal-demos

The code in `app.py` and `core/` could land in
`fal-demos/video/multi_perception_webrtc/` alongside `yolo_webcam_webrtc/`
and `matrix_webrtc/`. The frontend is a fork of
`yolo_webcam_webrtc/frontend` with three additions clearly diff-able:

1. A `.layer-toggles` button row in `index.html`
2. A `.hud` chip row in `index.html` + chip styling in `style.css`
3. A `LayerToggleInput` send path and a `type: "timing"` / `type: "runner_info"`
   handler in `main.js`

On the backend, `app.py` is ~150 lines of new code on top of a verbatim copy
of `yolo.py`'s signaling handler and Metered TURN bootstrap. The minimal diff
is: three Pydantic types added to the input/output unions, one handler
branch (`handle_layer`), one replacement of `create_yolo_track` with
`create_multi_perception_track`, and the extra model loads in `setup()`.

If fal wants to publish this as the next realtime example, the work is
renaming the namespace and writing two paragraphs of intro.

## Running it

### Option A — Local mode (any GPU machine / Mac)

No fal Serverless account required. The app runs entirely on the local
machine via `fal run --local`.

**Prerequisites**

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 18+
- Metered TURN credentials (`METERED_TURN_SECRET_KEY`, `METERED_TURN_LABEL`)

**Setup**

```bash
cp .env.example .env
# Edit .env and fill in your Metered TURN credentials.

uv sync              # installs fal + all runtime deps, generates uv.lock
```

**Backend** (Terminal 1)

```bash
uv run --env-file .env fal run --local app.py::MultiPerceptionWebRTC
```

The WebSocket server starts on `ws://localhost:8080/realtime`. Device is
auto-detected: CUDA on NVIDIA GPUs, MPS on Apple Silicon, CPU fallback.
YOLO weights are auto-downloaded on first run (~6 MB).

**Frontend** (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5173`). The "Local mode"
checkbox is on by default — just click **Start**.

**Remote GPU notes**: if running on a remote machine, expose ports 8080
(backend WS) and 5173 (Vite) and set the Backend URL field in the frontend
to `ws://<public-ip>:8080` instead of `ws://localhost:8080`.

**Apple Silicon notes**: runs on MPS. Expect slower inference than a
dedicated NVIDIA GPU but fully functional for development and testing.

### Option B — fal Serverless (cloud)

**Prerequisites**

- A fal account with Serverless access
- A Metered TURN account — hard-fails in `setup()` if
  `METERED_TURN_SECRET_KEY` or `METERED_TURN_LABEL` are not set
- `yolov8n.pt` available at `/data/yolov8n.pt` on the fal runner, or set
  `YOLO_MODEL_PATH` to a path Ultralytics can auto-download to

```bash
fal secrets set METERED_TURN_SECRET_KEY=<your-key>
fal secrets set METERED_TURN_LABEL=<your-label>
```

**Backend**

```bash
fal run app.py              # ephemeral, prints an app id you can hit
# or
fal deploy app.py           # persistent endpoint
```

**Frontend**

```bash
cd frontend
npm install
FAL_KEY=<your-fal-key> npm run dev
```

Uncheck "Local mode" in the UI, paste your app id (e.g.
`myuser/multi-perception-webrtc`) into the Endpoint field, and click
**Start**.
