import { decode, encode } from "@msgpack/msgpack";

const TOKEN_EXPIRATION_SECONDS = 120;
const DEFAULT_ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

const localModeCheckbox = document.getElementById("localMode");
const backendUrlInput = document.getElementById("backendUrl");
const backendUrlField = document.getElementById("backendUrlField");
const appIdField = document.getElementById("appIdField");
const appIdInput = document.getElementById("appId");
const usernameInput = document.getElementById("username");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const remoteVideoTitle = document.getElementById("remoteVideoTitle");
const localFpsEl = document.getElementById("localFps");
const remoteFpsEl = document.getElementById("remoteFps");
const localResEl = document.getElementById("localRes");
const remoteResEl = document.getElementById("remoteRes");
const localBitrateEl = document.getElementById("localBitrate");
const remoteBitrateEl = document.getElementById("remoteBitrate");
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");

const layerToggleEls = document.querySelectorAll(".layer-btn");
const hudRunnerEl = document.getElementById("hudRunner");
const hudModelsEl = document.getElementById("hudModels");
const hudYoloEl = document.getElementById("hudYolo");
const hudDepthEl = document.getElementById("hudDepth");
const hudSegEl = document.getElementById("hudSeg");
const hudTotalEl = document.getElementById("hudTotal");

const roomPanel = document.getElementById("roomPanel");
const roomActiveStatus = document.getElementById("roomActiveStatus");
const takeControlBtn = document.getElementById("takeControlBtn");
const participantsEl = document.getElementById("participants");

let ws = null;
let pc = null;
let started = false;
let authToken = "";
let localStream = null;
let localFpsStop = null;
let remoteFpsStop = null;
let bitrateStop = null;
let activeLayer = "detection";

let myPeerId = null;
let roomState = null;

const isLocalMode = () => localModeCheckbox.checked;

const syncModeFields = () => {
  backendUrlField.style.display = isLocalMode() ? "" : "none";
  appIdField.style.display = isLocalMode() ? "none" : "";
};
localModeCheckbox.addEventListener("change", syncModeFields);
syncModeFields();

const log = (msg) => {
  console.log(msg);
  logEl.textContent = String(msg) + "\n" + logEl.textContent;
};

const setStatus = (text) => {
  statusEl.textContent = text;
};

const normalizeAppId = (value) => value.replace(/^\/+|\/+$/g, "");

const buildWsUrl = (appId, token) => {
  const normalizedAppId = normalizeAppId(appId);
  return `wss://fal.run/${normalizedAppId}?fal_jwt_token=${encodeURIComponent(token)}`;
};

const getTemporaryAuthToken = async (appId) => {
  const response = await fetch("/fal/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      appId,
      tokenExpirationSeconds: TOKEN_EXPIRATION_SECONDS,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Token request failed: ${response.status} ${errorBody}`);
  }

  const token = await response.json();
  if (typeof token !== "string" && token?.detail) {
    return token.detail;
  }
  return token;
};

const resetHud = () => {
  hudRunnerEl.textContent = "runner: —";
  hudModelsEl.textContent = "models: —";
  hudYoloEl.textContent = "yolo: — ms";
  hudDepthEl.textContent = "depth: — ms";
  hudSegEl.textContent = "seg: — ms";
  hudTotalEl.textContent = "total: — ms";
};

const resetRoomUI = () => {
  roomPanel.style.display = "none";
  roomActiveStatus.textContent = "Waiting for players…";
  participantsEl.innerHTML = "";
  takeControlBtn.disabled = true;
  remoteVideoTitle.textContent = "Perception stack (processed)";
  myPeerId = null;
  roomState = null;
};

const stop = () => {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  if (localStream) {
    localStream.getTracks().forEach((track) => track.stop());
    localStream = null;
  }
  localVideo.srcObject = null;
  remoteVideo.srcObject = null;
  if (localFpsStop) {
    localFpsStop();
    localFpsStop = null;
  }
  if (remoteFpsStop) {
    remoteFpsStop();
    remoteFpsStop = null;
  }
  if (bitrateStop) {
    bitrateStop();
    bitrateStop = null;
  }
  localFpsEl.textContent = "FPS: --";
  remoteFpsEl.textContent = "FPS: --";
  localResEl.textContent = "-- x --";
  remoteResEl.textContent = "-- x --";
  localBitrateEl.textContent = "Up: -- Mbps";
  remoteBitrateEl.textContent = "Down: -- Mbps";
  resetHud();
  resetRoomUI();
  setStatus("Disconnected");
};

const parseIceServers = (iceServersPayload) => {
  if (!Array.isArray(iceServersPayload)) {
    return DEFAULT_ICE_SERVERS;
  }
  const servers = iceServersPayload.filter((item) => item && item.urls);
  return servers.length > 0 ? servers : DEFAULT_ICE_SERVERS;
};

const ensurePeer = async (iceServers) => {
  if (pc) return;
  pc = new RTCPeerConnection({ iceServers });

  pc.onconnectionstatechange = () => {
    log(`Peer connection state: ${pc.connectionState}`);
  };

  pc.oniceconnectionstatechange = () => {
    log(`ICE connection state: ${pc.iceConnectionState}`);
  };

  pc.onicecandidate = (event) => {
    if (event.candidate && ws) {
      sendWs({
        type: "icecandidate",
        candidate: {
          candidate: event.candidate.candidate,
          sdpMid: event.candidate.sdpMid,
          sdpMLineIndex: event.candidate.sdpMLineIndex,
        },
      });
    }
  };

  pc.ontrack = (event) => {
    log(`Track received: ${event.track?.kind || "unknown"}`);
    const stream = event.streams && event.streams[0]
      ? event.streams[0]
      : new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
    if (remoteFpsStop) {
      remoteFpsStop();
    }
    remoteFpsStop = startFpsCounter(remoteVideo, remoteFpsEl, remoteResEl);
  };
};

const attachLocalStream = async () => {
  if (localStream) return;
  localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  localVideo.srcObject = localStream;
  localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
  if (localFpsStop) {
    localFpsStop();
  }
  localFpsStop = startFpsCounter(localVideo, localFpsEl, localResEl);
  if (!bitrateStop) {
    bitrateStop = startBitrateMonitor();
  }
};

const startFpsCounter = (video, outputEl, resEl) => {
  let frameCount = 0;
  let lastTime = performance.now();
  let rafId = null;
  let stop = false;
  let lastRes = "";

  const update = (timestamp) => {
    if (stop) return;
    frameCount += 1;
    const delta = timestamp - lastTime;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width && height) {
      const nextRes = `${width} x ${height}`;
      if (nextRes !== lastRes) {
        resEl.textContent = nextRes;
        lastRes = nextRes;
      }
    }
    if (delta >= 1000) {
      const fps = Math.round((frameCount * 1000) / delta);
      outputEl.textContent = `FPS: ${fps}`;
      frameCount = 0;
      lastTime = timestamp;
    }
    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback((_, metadata) => update(metadata.expectedDisplayTime || performance.now()));
    } else {
      rafId = requestAnimationFrame(update);
    }
  };

  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback((_, metadata) => update(metadata.expectedDisplayTime || performance.now()));
  } else {
    rafId = requestAnimationFrame(update);
  }

  return () => {
    stop = true;
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
  };
};

const startBitrateMonitor = () => {
  let lastOutBytes = null;
  let lastOutTime = null;
  let lastInBytes = null;
  let lastInTime = null;

  const update = async () => {
    if (!pc) return;
    const stats = await pc.getStats();
    stats.forEach((report) => {
      if (report.type === "outbound-rtp" && report.kind === "video") {
        if (lastOutBytes !== null && lastOutTime !== null) {
          const bytes = report.bytesSent;
          const time = report.timestamp;
          const bitrate = ((bytes - lastOutBytes) * 8) / ((time - lastOutTime) / 1000);
          localBitrateEl.textContent = `Up: ${Math.max(0, bitrate / 1_000_000).toFixed(2)} Mbps`;
        }
        lastOutBytes = report.bytesSent;
        lastOutTime = report.timestamp;
      }
      if (report.type === "inbound-rtp" && report.kind === "video") {
        if (lastInBytes !== null && lastInTime !== null) {
          const bytes = report.bytesReceived;
          const time = report.timestamp;
          const bitrate = ((bytes - lastInBytes) * 8) / ((time - lastInTime) / 1000);
          remoteBitrateEl.textContent = `Down: ${Math.max(0, bitrate / 1_000_000).toFixed(2)} Mbps`;
        }
        lastInBytes = report.bytesReceived;
        lastInTime = report.timestamp;
      }
    });
  };

  const intervalId = setInterval(() => {
    update().catch((err) => log(`Bitrate stats error: ${err.message || err}`));
  }, 1000);

  return () => clearInterval(intervalId);
};

const sendOffer = async () => {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  sendWs({ type: "offer", sdp: offer.sdp });
};

const sendWs = (payload) => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const encoded = encode(payload);
  const buffer = encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  );
  ws.send(buffer);
};

const readWsMessage = async (data) => {
  if (data instanceof ArrayBuffer) {
    return decode(new Uint8Array(data));
  }
  if (data instanceof Blob) {
    const buffer = await data.arrayBuffer();
    return decode(new Uint8Array(buffer));
  }
  if (typeof data === "string") {
    try {
      return JSON.parse(data);
    } catch {
      return data;
    }
  }
  return data;
};

const fmtMs = (v) => (v == null ? "—" : `${v.toFixed(0)}`);

const applyTimingMessage = (msg) => {
  hudYoloEl.textContent = `yolo: ${fmtMs(msg.yolo_ms)} ms`;
  hudDepthEl.textContent = `depth: ${fmtMs(msg.depth_ms)} ms`;
  hudSegEl.textContent = `seg: ${fmtMs(msg.seg_ms)} ms`;
  hudTotalEl.textContent = `total: ${fmtMs(msg.total_ms)} ms`;
};

const applyRunnerInfo = (msg) => {
  hudRunnerEl.textContent = `runner: ${msg.runner_id} · warm`;
  hudModelsEl.textContent = `models: ${(msg.models || []).length} loaded`;
};

// ---- Room UI ----

const updateRoomUI = (state) => {
  roomState = state;
  roomPanel.style.display = "";

  const activePeer = state.peers.find((p) => p.is_active);
  const iAmActive = activePeer && activePeer.peer_id === myPeerId;

  if (iAmActive) {
    roomActiveStatus.textContent = "You are the active camera";
    roomActiveStatus.classList.add("you-active");
    takeControlBtn.disabled = true;
  } else if (activePeer) {
    roomActiveStatus.textContent = `Watching: ${activePeer.username}`;
    roomActiveStatus.classList.remove("you-active");
    takeControlBtn.disabled = false;
  } else {
    roomActiveStatus.textContent = "No active camera";
    roomActiveStatus.classList.remove("you-active");
    takeControlBtn.disabled = false;
  }

  if (activePeer) {
    remoteVideoTitle.textContent = iAmActive
      ? "Your processed feed (active)"
      : `${activePeer.username}'s processed feed`;
  } else {
    remoteVideoTitle.textContent = "Perception stack (processed)";
  }

  participantsEl.innerHTML = state.peers
    .map((p) => {
      const isMe = p.peer_id === myPeerId;
      const classes = [
        "participant",
        p.is_active ? "active" : "",
        isMe ? "me" : "",
      ]
        .filter(Boolean)
        .join(" ");
      const label = isMe ? `${p.username} (you)` : p.username;
      const badge = p.is_active ? '<span class="active-badge">LIVE</span>' : "";
      return `<div class="${classes}">${badge}<span class="participant-name">${label}</span></div>`;
    })
    .join("");
};

takeControlBtn.addEventListener("click", () => {
  sendWs({ type: "take_control" });
});

// ---- Layer toggles ----

const setActiveLayer = (layer) => {
  activeLayer = layer;
  layerToggleEls.forEach((btn) => {
    if (btn.dataset.layer === layer) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  sendWs({ type: "layer", layer });
};

layerToggleEls.forEach((btn) => {
  btn.addEventListener("click", () => {
    setActiveLayer(btn.dataset.layer);
  });
});

// ---- Connection ----

const buildLocalWsUrl = () => {
  const base = backendUrlInput.value.trim().replace(/\/+$/, "");
  return `${base}/realtime`;
};

const getUsername = () => {
  const val = usernameInput.value.trim();
  return val || `Player-${Math.random().toString(36).slice(2, 6)}`;
};

const connectWs = (wsUrl) => {
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  let pendingIceServers = DEFAULT_ICE_SERVERS;
  let offerSent = false;
  let negotiatingOffer = false;

  const ensureOfferSent = async () => {
    if (offerSent || negotiatingOffer) return;
    negotiatingOffer = true;
    offerSent = true;
    try {
      await ensurePeer(pendingIceServers);
      await attachLocalStream();
      await sendOffer();
      sendWs({ type: "layer", layer: activeLayer });
    } catch (err) {
      offerSent = false;
      throw err;
    } finally {
      negotiatingOffer = false;
    }
  };

  ws.onopen = async () => {
    setStatus("Connected");
    log("WebSocket open.");
    sendWs({ type: "join", username: getUsername() });
  };

  ws.onmessage = async (event) => {
    const msg = await readWsMessage(event.data);
    if (!msg || typeof msg !== "object") {
      log(`WS: ${String(msg)}`);
      return;
    }

    if (msg.type === "joined") {
      myPeerId = msg.peer_id;
      log(`Joined room as peer ${myPeerId}`);
    } else if (msg.type === "room_state") {
      updateRoomUI(msg);
      log(`Room: ${msg.peers.length} player(s), active: ${msg.active_peer_id || "none"}`);
    } else if (msg.type === "iceservers" && !offerSent) {
      pendingIceServers = parseIceServers(msg.iceservers);
      log(`Using ${pendingIceServers.length} ICE server entries from signaling.`);
      try {
        await ensureOfferSent();
      } catch (err) {
        log(`Failed to get webcam: ${err.message || err}`);
        stop();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        started = false;
        return;
      }
    } else if (msg.type === "runner_info") {
      applyRunnerInfo(msg);
      log(`Runner ${msg.runner_id} reports ${(msg.models || []).join(", ")}`);
    } else if (msg.type === "timing") {
      applyTimingMessage(msg);
    } else if (msg.type === "answer" && msg.sdp && pc) {
      await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: msg.sdp }));
    } else if (msg.type === "icecandidate" && pc) {
      const c = msg.candidate;
      if (c) {
        await pc.addIceCandidate(new RTCIceCandidate({
          candidate: c.candidate,
          sdpMid: c.sdpMid,
          sdpMLineIndex: c.sdpMLineIndex,
        }));
      }
    } else if (msg.type === "error") {
      log(`Server error: ${msg.error}`);
    } else {
      log(`WS message: ${JSON.stringify(msg)}`);
    }
  };

  ws.onclose = (ev) => {
    log(`WebSocket closed code=${ev.code} reason=${ev.reason}`);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  };

  ws.onerror = (err) => {
    log("WebSocket error.");
    console.error(err);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  };
};

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  logEl.textContent = "";
  resetHud();
  resetRoomUI();

  if (isLocalMode()) {
    const wsUrl = buildLocalWsUrl();
    log(`Local mode → ${wsUrl}`);
    connectWs(wsUrl);
    return;
  }

  const appId = normalizeAppId(appIdInput.value.trim());
  if (!appId) {
    log("Missing endpoint.");
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

  try {
    authToken = await getTemporaryAuthToken(appId);
  } catch (err) {
    log(`Failed to fetch token: ${err.message || err}`);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

  const wsUrl = buildWsUrl(appId, authToken);
  connectWs(wsUrl);
});

stopBtn.addEventListener("click", () => {
  stop();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  started = false;
});
