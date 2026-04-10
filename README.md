# Face Swap Arena — Multiplayer WebRTC on fal Serverless

A multiplayer `fal.App` that streams live webcam video from multiple peers through
a server-side AI pipeline — face detection, face swapping, and optional enhancement —
and broadcasts the processed composite grid back to every participant in real time.

## What it does

- Multiple users join a shared room over WebRTC
- Each peer's video is captured server-side and faces are extracted per-frame
- A reference face can be uploaded by any user; the server swaps all detected faces to match it
- **Shuffle mode**: randomly reassigns captured peer faces among participants (no one gets their own — uses Sattolo's algorithm)
- **Enhancement toggle**: applies GFPGAN face enhancement to swapped faces
- **YOLO overlay**: optional YOLOv8n object detection drawn on top of the stream
- A live performance HUD shows per-model latency, runner ID, current mode, and active models

## Why this shape is the right one for production realtime vision

Real-time multi-user face processing has a hard **per-frame latency budget**. Any
network round-trip between separate model runners — even on the same datacenter —
adds 30–60 ms per hop. Face detection → face swap → enhancement as three separate
marketplace calls would blow the ~80 ms budget before the first frame arrives.

Co-locating every model in `setup()` on one warm runner eliminates those hops.
All inference runs sequentially on the same GPU with zero network overhead.
fal Serverless is the only platform where that pattern is clean and deployable
in a single file.

## What's here

```
app.py                 # fal.App — WebSocket signaling, room lifecycle, message types
pyproject.toml         # uv project with all runtime deps
.env.example           # template for Metered TURN credentials
core/
  perception.py        # YOLOv8n wrapper
  face_swap.py         # InsightFace + inswapper + GFPGAN pipeline
  room.py              # Room, PeerState, FrameBroadcaster, processing loop
frontend/
  package.json         # self-contained Vite project
  vite.config.js       # dev server config
  index.html           # join overlay, video grid, controls, HUD, debug panel
  src/main.js          # WebRTC signaling, face upload, room state, shuffle UI
  src/style.css        # grid layout, toggle buttons, HUD chips, shuffle flash
```

## Design decisions

- **All models co-located in `setup()`** — each is warmed with a black frame before
  the first real peer connects, keeping CUDA JIT cost off the user-visible budget.

- **Sequential inference, not `asyncio.gather`** — all models are CUDA-bound on the
  same GPU and would serialize on the same stream under gather anyway. Sequential
  is honest. Total budget is well within the H100 frame budget.

- **Face capture is async** — face extraction from peer frames runs off the
  hot path so it doesn't block the broadcast loop.

- **`LatestFrameRelay` drains track queues** — only the most recent frame from
  each peer is kept. This prevents frame accumulation lag when multiple peers
  are connected and processing falls momentarily behind ingestion.

- **Sattolo's algorithm for shuffle** — guarantees no peer ever receives their own
  face, which is the only shuffle outcome that would break the illusion.

- **Room state broadcast** — a single `room_state` message is pushed to all peers
  whenever membership or face assignment changes. This is the single source of
  truth for the participant list and current face assignments.

- **MessagePack over JSON** for the signaling WebSocket — more compact for binary
  payloads (SDP, face thumbnails, msgpack frames).

- **`local_python_modules = ["core"]`** packages the local `core/` directory onto
  the runner. Same pattern as `fal_demos/image/sana.py`.

- **`machine_type = "GPU-H100"`** — all three models fit in ~450 MB of VRAM
  (< 1 % of the H100's 80 GB), leaving headroom for larger crowds or additional models.

## VRAM budget

| Model                  | Approx. VRAM |
| ---------------------- | ------------ |
| YOLOv8n                | ~160 MB      |
| InsightFace (buffalo_l)| ~160 MB      |
| inswapper_128_fp16     | ~128 MB      |
| GFPGAN (optional)      | loaded on demand |
| **Total (no enhance)** | **~450 MB**  |

~450 MB on an 80 GB H100 is < 1 %. The co-location pattern has no resource
conflict — it is strictly cheaper than calling separate marketplace endpoints
per frame because the per-frame network cost of three hops swamps the idle cost
of keeping one runner warm.

## What this isn't

- **Not a multi-GPU demo.** For multi-GPU, use `fal.distributed`. This demo is
  about co-locating several models on one GPU.
- **Not optimized for max framerate.** TensorRT or ONNX conversions would reduce
  per-frame budget further. That's a hardening pass.
- **Not production-hardened.** No auth, no rate limiting, no observability beyond
  `print`s. A real deployment would add all three.

## Running it

### Prerequisites (both options)

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 18+
- Metered TURN credentials (`METERED_TURN_SECRET_KEY`, `METERED_TURN_LABEL`)

### Option A — Local mode

No fal account required. The app runs entirely on the local machine via `fal run --local`.

```bash
cp .env.example .env
# Edit .env and fill in your Metered TURN credentials.

uv sync
```

**Backend** (Terminal 1)

```bash
uv run --env-file .env fal run --local app.py::MultiPerceptionWebRTC
```

The WebSocket server starts on `ws://localhost:8080/realtime`. Device is
auto-detected: CUDA on NVIDIA GPUs, MPS on Apple Silicon, CPU fallback.
YOLO and InsightFace weights are downloaded on first run.

**Frontend** (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Enter a username and click **Join** to enter the room.

**Remote GPU notes**: expose ports 8080 and 5173, set the Backend URL in the
frontend to `ws://<public-ip>:8080`.

**Apple Silicon notes**: runs on MPS. Inference is slower than a dedicated NVIDIA
GPU but fully functional for development and testing.

### Option B — fal Serverless (cloud)

```bash
fal secrets set METERED_TURN_SECRET_KEY=<your-key>
fal secrets set METERED_TURN_LABEL=<your-label>

fal run app.py        # ephemeral
# or
fal deploy app.py     # persistent endpoint
```

**Frontend**

```bash
cd frontend
npm install
FAL_KEY=<your-fal-key> npm run dev
```

Paste your app endpoint into the Backend URL field in the UI and click **Join**.

## Using the app

1. **Join** — enter a username and click Join. Your local video appears immediately.
2. **Upload a face** — drag a photo onto the face upload zone or click to browse.
   InsightFace detects the largest face and stores it as your source. All faces
   in the broadcast stream are swapped to match it.
3. **Enhancement** — toggle the Enhance switch to apply GFPGAN to swapped faces.
4. **Shuffle** — click Shuffle in the room panel to randomly reassign all captured
   peer faces. Each participant gets someone else's face.
5. **YOLO** — toggle Detection to overlay YOLOv8n bounding boxes on the stream.
6. **Clear face** — click the × on the face upload zone to remove your source face
   and stop swapping.
