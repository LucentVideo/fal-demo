# Lucent Serverless — Architecture

A serverless realtime platform on RunPod. Users write a Python class,
run one command, and get a live websocket endpoint backed by a warm pod.
When nobody's connected, the pod terminates and billing stops.

---

## Components

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   User SDK   │ ──────► │   Controller     │ ──────► │   Worker Pod     │
│  (client)    │         │ (persistent pod) │         │  (RunPod, ephemeral)
└──────────────┘         └──────────────────┘         └──────────────────┘
  lucent-serverless        lucent-serverless-controller    lucent-base-cpu/gpu
```

### 1. User SDK (`lucent-serverless`)

Python package the user installs. Contains:

- **`App` base class** — user subclasses this, sets attributes (`app_id`,
  `compute_type`, `keep_alive`, etc.), overrides `setup()`, and decorates
  methods with `@realtime("/path")`.
- **`deploy()`** — registers app config with the controller (POST /apps) and
  uploads the user's code directory as a tarball (PUT /apps/{id}/code).
- **`resolve()`** — asks the controller for a warm pod URL. Cold-starts one
  if needed.
- **`connect()`** — resolve + open websocket in one call.
- **`App.spawn()`** — deploy + upload code + resolve in one call. The
  recommended way to use the SDK.
- **CLI: `lucent-deploy`** — deploy from the command line without writing
  client code.

When `image_ref` is not set on the App class, the SDK defaults to the
platform base image (`lucent-base-cpu` or `lucent-base-gpu`).

### 2. Controller (`lucent-serverless-controller`)

A persistent CPU pod on RunPod. The brain of the system. Runs FastAPI with
three background concerns:

- **Scheduler thread** — polls `/health` on every live pod every 5 seconds.
  Updates connection counts and activity timestamps. Records boot timings
  into `pod_history`. Ensures `min_concurrency` by spawning pods if needed.
- **Reaper thread** — checks idle pods every 30 seconds. If a pod has been
  idle longer than `keep_alive_sec`, it terminates the RunPod pod and
  updates `pod_history` with `terminated_at`.
- **Code store** — stores user app code tarballs on disk. Pods download
  their code from the controller at boot via `GET /apps/{id}/code`.

**Key endpoints:**

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /apps` | yes | Register/update app config |
| `PUT /apps/{id}/code` | yes | Upload app code tarball |
| `GET /apps/{id}/code` | no | Download code (pods fetch this) |
| `GET /resolve?app_id=X` | yes | Get a pod URL (cold-starts if needed) |
| `GET /pod-history` | yes | Pod lifecycle data + boot timings |
| `GET /dashboard` | no | Admin dashboard |

**Database (SQLite):**

- `apps` — app config (image, compute type, scaling, keep alive)
- `pods` — live pod state (status, connections, health poll data)
- `pod_history` — full lifecycle of every pod ever spawned (created,
  boot timings, terminated, cold start duration)

Thread safety: one shared connection with `check_same_thread=False` and a
`threading.Lock` around all writes. WAL mode for concurrent reads.

### 3. Worker Pod (base images)

Ephemeral pods on RunPod. Boot from a shared base image, download user
code at startup, then run the user's app.

**Base images:**

- `lucent-base-cpu` — Python 3.12 + uv + curl + lucent runner
- `lucent-base-gpu` — (planned) RunPod PyTorch base + common ML deps + runner

**Boot sequence (entrypoint.sh):**

```
1. Record boot_start timestamp
2. curl code tarball from controller → tar xz into /app
3. Record code_downloaded timestamp
4. uv pip install -r requirements.txt (if present)
5. Record deps_installed timestamp
6. python -m lucent_serverless.runner
   → imports user module (LUCENT_APP_MODULE env var)
   → finds App subclass
   → runs setup()                    ← setup_start/setup_done
   → mounts @realtime websocket routes
   → starts uvicorn on :8000         ← ready_at
```

**Runner (`lucent_serverless.runner`):**

- Imports the user's module, finds the `App` subclass
- Calls `setup()` (where users load models)
- Walks methods for `@realtime` decorators, mounts them as FastAPI
  websocket routes
- Exposes `/health` with connection count, last activity, and boot timings
- The worker never self-terminates — that's the reaper's job

---

## Data Flow

### Deploy + Spawn

```
python app.py
  │
  ├─ deploy()
  │    POST /apps {config}           → controller upserts into apps table
  │
  ├─ upload_code()
  │    tar.gz the app directory
  │    PUT /apps/echo/code           → controller saves to disk
  │
  └─ resolve("echo")
       GET /resolve?app_id=echo      → controller checks for warm pods
       │                                none found → cold start:
       │
       ├─ controller calls RunPod REST API: POST /v1/pods
       │    creates pod with image=lucent-base-cpu
       │    env: LUCENT_CODE_URL=https://controller/apps/echo/code
       │
       ├─ controller inserts into pods table + pod_history
       │
       ├─ controller polls pod /health until 200
       │    (RunPod schedules → pulls image → starts container →
       │     entrypoint downloads code → runner starts → /health 200)
       │
       └─ returns pod_url to client
```

### Websocket Connection

```
async with ls.connect("echo", "/realtime") as ws:
    await ws.send("hello")       →  wss://pod-8000.proxy.runpod.net/realtime
    print(await ws.recv())       ←  "echo: hello"
```

The RunPod proxy terminates TLS. The pod's FastAPI websocket handler
receives the connection, increments the connection counter, calls the
user's `@realtime` method, and decrements on disconnect.

### Pod Lifecycle Tracking

Every pod gets a row in `pod_history`:

```
created_at          ← controller calls RunPod API
boot_start          ← entrypoint.sh starts (container running)
code_downloaded     ← tarball extracted
deps_installed      ← uv pip install done (if requirements.txt)
setup_start         ← App.setup() begins
setup_done          ← App.setup() returns
ready_at            ← uvicorn listening, /health returns 200
terminated_at       ← reaper kills the pod

total_boot_sec      = ready_at - boot_start     (in-container time)
total_cold_start_sec = ready_at - created_at     (user-perceived wait)
```

The dashboard shows this as a timing breakdown bar per pod:

```
 ┌──────────┬────┬──────┬───┬───┐
 │  infra   │code│ deps │ s │ u │  4.2s total
 └──────────┴────┴──────┴───┴───┘
  RunPod      dl   pip   setup uvicorn
```

---

## Key Design Decisions

**No Docker builds for app changes.** Users don't build images. They write
Python, the SDK tars it up and uploads it. The base image is shared. This
makes iteration seconds instead of minutes.

**Controller stores code.** Simplest option — no S3, no registry. Tarballs
live on disk next to the SQLite DB. Pods curl them at boot. Scales to
hundreds of apps easily; move to object storage later if needed.

**SQLite with write lock.** One shared connection, one `threading.Lock` for
writes. SQLite only allows one writer at a time; per-thread connections
caused "database is locked" errors even with WAL mode. The write lock
serializes scheduler, reaper, and API handler writes cleanly.

**Worker never self-terminates.** RunPod's container restart policy just
relaunches the process, burning money in a loop. Pod termination is
always the reaper's job (DELETE /v1/pods from outside).

**uv for dependency installs.** 10-50x faster than pip. Installed in the
base image. User's `requirements.txt` gets installed at boot, not at
image build time.

---

## Repository Layout

```
lucent-serverless/                    # SDK + worker package
  lucent_serverless/
    __init__.py                       # App, deploy, resolve, connect, upload_code
    runner.py                         # Generic entrypoint for worker pods
    state.py                          # Connection tracking + boot timings
    runpod_client.py                  # RunPod REST API client
    cli/
      deploy.py                       # lucent-deploy CLI
      spawn.py                        # lucent-spawn CLI
      terminate.py                    # lucent-terminate CLI
  images/
    cpu/
      Dockerfile                      # lucent-base-cpu image
      entrypoint.sh                   # Boot script with timing
  examples/
    echo/
      app.py                          # Minimal echo app

lucent-serverless-controller/         # Control plane
  lucent_serverless_controller/
    main.py                           # FastAPI app, lifespan, auth
    db.py                             # SQLite: apps, pods, pod_history
    resolver.py                       # /resolve, app CRUD, pod-history endpoints
    scheduler.py                      # Health polling, pod spawning
    reaper.py                         # Idle pod termination
    code_store.py                     # Code upload/download endpoints
    dashboard.html                    # Admin dashboard
```
