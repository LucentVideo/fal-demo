import { decode, encode } from "@msgpack/msgpack";

const TOKEN_EXPIRATION_SECONDS = 120;
const DEFAULT_ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

// Auto-detect WebSocket URL from page location (works on RunPod proxy)
const _autoWsUrl = () => {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
};
const ENV_BACKEND_URL = import.meta.env.VITE_BACKEND_URL || _autoWsUrl();
const ENV_LOCAL_MODE = import.meta.env.VITE_LOCAL_MODE === "true";
// Stable lucent controller URL (https://). When set, the frontend asks the
// controller to resolve the current pod for this app_id instead of going
// through fal. Pod URLs change per cold-start, but the controller URL
// doesn't — so this env var is safe to bake into the build.
const ENV_LUCENT_CONTROLLER_URL = import.meta.env.VITE_LUCENT_CONTROLLER_URL || "";

const localModeCheckbox = document.getElementById("localMode");
const backendUrlInput = document.getElementById("backendUrl");
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
const hudModeEl = document.getElementById("hudMode");
const hudTotalEl = document.getElementById("hudTotal");

const faceFileInput = document.getElementById("faceFileInput");
const faceUploadZone = document.getElementById("faceUploadZone");
const facePreview = document.getElementById("facePreview");
const faceUploadPrompt = document.getElementById("faceUploadPrompt");
const clearFaceBtn = document.getElementById("clearFaceBtn");
const enhanceToggle = document.getElementById("enhanceToggle");
const faceStatusEl = document.getElementById("faceStatus");

const roomPanel = document.getElementById("roomPanel");
const roomStatusEl = document.getElementById("roomStatus");
const participantsEl = document.getElementById("participants");
const shuffleBtn = document.getElementById("shuffleBtn");
const shuffleStatusEl = document.getElementById("shuffleStatus");
const debugSection = document.getElementById("debugSection");
const debugToggle = document.getElementById("debugToggle");
const shuffleFlash = document.getElementById("shuffleFlash");
const joinOverlay = document.getElementById("joinOverlay");
const joinLoadingEl = document.getElementById("joinLoading");
const loadingStageEl = document.getElementById("loadingStage");
const loadingHintEl = document.getElementById("loadingHint");
const videoLoadingEl = document.getElementById("videoLoading");
const videoLoadingStageEl = document.getElementById("videoLoadingStage");

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
let shuffleAssignments = null;

localModeCheckbox.value = ENV_LOCAL_MODE ? "true" : "";
backendUrlInput.value = ENV_BACKEND_URL;

const isLocalMode = () => ENV_LOCAL_MODE;

const _t0 = performance.now();
const tlog = (msg) => {
  const elapsed = (performance.now() - _t0).toFixed(0);
  const stamped = `[+${elapsed}ms] ${msg}`;
  console.log(stamped);
  logEl.textContent = stamped + "\n" + logEl.textContent;
};
const log = tlog;

const setStatus = (text) => {
  statusEl.textContent = text;
};

// ---- Loading UI ----
// showJoinLoading() swaps the Join button out for a spinner + staged status.
// showVideoLoading() covers the video stage after the overlay hides, until
// the first remote frame arrives. Both hide via clearLoading().
const showJoinLoading = (stage, hint = "") => {
  startBtn.hidden = true;
  joinLoadingEl.hidden = false;
  loadingStageEl.textContent = stage;
  loadingHintEl.textContent = hint;
};

const updateJoinLoading = (stage, hint) => {
  if (joinLoadingEl.hidden) return;
  if (stage !== undefined) loadingStageEl.textContent = stage;
  if (hint !== undefined) loadingHintEl.textContent = hint;
};

const showVideoLoading = (stage) => {
  videoLoadingEl.hidden = false;
  if (stage) videoLoadingStageEl.textContent = stage;
};

const updateVideoLoading = (stage) => {
  if (!videoLoadingEl.hidden && stage) videoLoadingStageEl.textContent = stage;
};

const clearLoading = () => {
  startBtn.hidden = false;
  joinLoadingEl.hidden = true;
  loadingStageEl.textContent = "";
  loadingHintEl.textContent = "";
  videoLoadingEl.hidden = true;
};

const normalizeAppId = (value) => value.replace(/^\/+|\/+$/g, "");

const buildWsUrl = (appId, token) => {
  const normalizedAppId = normalizeAppId(appId);
  return `wss://fal.run/${normalizedAppId}?fal_jwt_token=${encodeURIComponent(token)}`;
};

const resolveLucentPodWsUrl = async (appId) => {
  const base = ENV_LUCENT_CONTROLLER_URL.replace(/\/+$/, "");
  const startedAt = performance.now();
  const deadline = startedAt + 600_000; // 10 min
  let lastStatus = null;
  while (true) {
    const resp = await fetch(
      `${base}/resolve?app_id=${encodeURIComponent(appId)}`,
      { method: "GET" },
    );
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`resolve failed (${resp.status}): ${body}`);
    }
    const data = await resp.json();
    const status = data.status || "ready";
    if (status === "ready") {
      const podUrl = data.pod_url.replace(/^https:/, "wss:").replace(/^http:/, "ws:");
      log(`resolved pod ${data.pod_id} in ${(performance.now() - startedAt).toFixed(0)}ms`);
      updateJoinLoading("Pod ready — connecting…", "");
      return `${podUrl.replace(/\/+$/, "")}/realtime`;
    }
    if (status !== lastStatus) {
      log(`pod ${data.pod_id || "?"} status=${status}, waiting…`);
      lastStatus = status;
      if (status === "pending") {
        updateJoinLoading(
          "Waking up GPU pod…",
          "Cold starts can take 2–3 minutes. Once warm, future joins are instant.",
        );
      }
    }
    if (performance.now() > deadline) {
      throw new Error(`pod for ${appId} did not become ready in time`);
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
};

const getTemporaryAuthToken = async (appId) => {
  log("Fetching fal auth token…");
  const t = performance.now();
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
  log(`Token fetched in ${(performance.now() - t).toFixed(0)}ms`);
  if (typeof token !== "string" && token?.detail) {
    return token.detail;
  }
  return token;
};

// Prefetch token on page load so it's ready by the time the user clicks Join.
let _prefetchedToken = null;
let _prefetchedTokenExpires = 0;
const _prefetchToken = () => {
  if (isLocalMode() || ENV_LUCENT_CONTROLLER_URL) return;
  const appId = normalizeAppId(appIdInput.value.trim());
  if (!appId) return;
  getTemporaryAuthToken(appId)
    .then((tok) => {
      _prefetchedToken = tok;
      _prefetchedTokenExpires = Date.now() + (TOKEN_EXPIRATION_SECONDS - 10) * 1000;
      log("Token prefetched (ready for instant join)");
    })
    .catch(() => { /* will retry on click */ });
};
setTimeout(_prefetchToken, 200);

const resetHud = () => {
  hudRunnerEl.textContent = "runner: —";
  hudModelsEl.textContent = "models: —";
  hudModeEl.textContent = "mode: —";
  hudTotalEl.textContent = "total: — ms";
};

const applyFaceOverrideUI = (active, setBy, imageData) => {
  if (active) {
    faceStatusEl.textContent = `Override active (by ${setBy})`;
    faceStatusEl.className = "face-status success";
    clearFaceBtn.disabled = false;
    if (imageData) {
      facePreview.src = imageData;
      facePreview.style.display = "";
      faceUploadPrompt.style.display = "none";
    }
  } else {
    faceStatusEl.textContent = "";
    faceStatusEl.className = "face-status";
    clearFaceBtn.disabled = true;
    facePreview.style.display = "none";
    facePreview.src = "";
    faceUploadPrompt.style.display = "";
  }
};

const resetRoomUI = () => {
  roomPanel.style.display = "none";
  roomStatusEl.textContent = "Waiting for friends\u2026";
  participantsEl.innerHTML = "";
  remoteVideoTitle.textContent = "Everyone";
  myPeerId = null;
  roomState = null;
  shuffleAssignments = null;
  shuffleBtn.disabled = true;
  shuffleStatusEl.textContent = "";
  applyFaceOverrideUI(false);
};

const updateShuffleBtnState = () => {
  if (!roomState) {
    shuffleBtn.disabled = true;
    return;
  }
  const capturedCount = roomState.peers.filter((p) => p.face_captured).length;
  shuffleBtn.disabled = capturedCount < 2;
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
  setStatus("");
  clearLoading();
  joinOverlay.classList.remove("hidden");
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
    updateVideoLoading("Buffering first frame…");
    const stream = event.streams && event.streams[0]
      ? event.streams[0]
      : new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
    // Clear the spinner once the first frame actually paints.
    remoteVideo.addEventListener(
      "playing",
      () => { videoLoadingEl.hidden = true; },
      { once: true },
    );
    if (remoteFpsStop) {
      remoteFpsStop();
    }
    remoteFpsStop = startFpsCounter(remoteVideo, remoteFpsEl, remoteResEl);
  };
};

// Pre-acquire camera early so getUserMedia permission dialog doesn't block the join flow.
let _earlyStreamPromise = null;
const _warmCamera = () => {
  if (_earlyStreamPromise) return;
  _earlyStreamPromise = navigator.mediaDevices
    .getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
    .catch(() => null);
};

const attachLocalStream = async () => {
  if (localStream) return;
  localStream = (_earlyStreamPromise && await _earlyStreamPromise) ||
    await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
  _earlyStreamPromise = null;
  localVideo.srcObject = localStream;
  localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

  // Request higher bitrate from the WebRTC encoder
  try {
    const senders = pc.getSenders().filter((s) => s.track?.kind === "video");
    for (const sender of senders) {
      const params = sender.getParameters();
      if (!params.encodings || params.encodings.length === 0) {
        params.encodings = [{}];
      }
      params.encodings[0].maxBitrate = 2_500_000; // 2.5 Mbps
      await sender.setParameters(params);
    }
  } catch (e) {
    console.warn("Could not set sender bitrate:", e);
  }

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
  hudModeEl.textContent = `mode: ${msg.layer || "—"}`;
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

  const count = state.peers.length;
  roomStatusEl.textContent = `${count} ${count !== 1 ? "people" : "person"} here`;
  remoteVideoTitle.textContent =
    count > 0 ? `Everyone (${count})` : "Everyone";

  if (state.face_override_active) {
    applyFaceOverrideUI(true, state.face_override_by, state.face_override_image || null);
  }

  participantsEl.innerHTML = state.peers
    .map((p) => {
      const isMe = p.peer_id === myPeerId;
      const classes = ["participant", isMe ? "me" : ""].filter(Boolean).join(" ");
      const label = isMe ? `${p.username} (you)` : p.username;
      const videoDot = p.has_video ? '<span class="video-dot"></span>' : "";
      const faceIcon = p.face_captured
        ? '<span class="face-captured-icon" title="Face captured">&#9786;</span>'
        : '<span class="face-pending-icon" title="Detecting face…">&#9676;</span>';
      let assignmentTag = "";
      if (shuffleAssignments) {
        const assignment = shuffleAssignments.find((a) => a.peer_id === p.peer_id);
        if (assignment) {
          assignmentTag = `<span class="face-assignment">${assignment.assigned_face_of}'s face</span>`;
        }
      }
      return `<div class="${classes}">${videoDot}${faceIcon}<span class="participant-name">${label}</span>${assignmentTag}</div>`;
    })
    .join("");

  updateShuffleBtnState();
};

// ---- Face swap controls ----

const uploadFaceImage = (file) => {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    facePreview.src = dataUrl;
    facePreview.style.display = "";
    faceUploadPrompt.style.display = "none";
    sendWs({ type: "set_source_face", image_data: dataUrl });
    faceStatusEl.textContent = "Uploading…";
  };
  reader.readAsDataURL(file);
};

faceUploadZone.addEventListener("click", () => faceFileInput.click());
faceFileInput.addEventListener("change", () => {
  if (faceFileInput.files.length > 0) uploadFaceImage(faceFileInput.files[0]);
});

faceUploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  faceUploadZone.classList.add("drag-over");
});
faceUploadZone.addEventListener("dragleave", () => {
  faceUploadZone.classList.remove("drag-over");
});
faceUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  faceUploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) uploadFaceImage(e.dataTransfer.files[0]);
});

clearFaceBtn.addEventListener("click", () => {
  sendWs({ type: "clear_source_face" });
});

enhanceToggle.addEventListener("change", () => {
  sendWs({ type: "toggle_enhance", enabled: enhanceToggle.checked });
});

shuffleBtn.addEventListener("click", () => {
  sendWs({ type: "shuffle_faces" });
  shuffleStatusEl.textContent = "Shuffling\u2026";
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
      let t = performance.now();
      await ensurePeer(pendingIceServers);
      log(`RTCPeerConnection created in ${(performance.now() - t).toFixed(0)}ms`);
      t = performance.now();
      await attachLocalStream();
      log(`getUserMedia + addTrack in ${(performance.now() - t).toFixed(0)}ms`);
      t = performance.now();
      await sendOffer();
      log(`Offer sent in ${(performance.now() - t).toFixed(0)}ms`);
      sendWs({ type: "layer", layer: activeLayer });
    } catch (err) {
      offerSent = false;
      throw err;
    } finally {
      negotiatingOffer = false;
    }
  };

  ws.onopen = async () => {
    setStatus("Connecting\u2026");
    log("WebSocket open.");
    // Join overlay fades out; swap to the video-stage spinner until the
    // first remote frame actually paints (WebRTC negotiation takes a
    // couple seconds on top of the ws handshake).
    joinOverlay.classList.add("hidden");
    joinLoadingEl.hidden = true;
    showVideoLoading("Negotiating video stream…");
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
      log(`Room: ${msg.peers.length} player(s)`);
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
    } else if (msg.type === "source_face_set") {
      if (msg.success && msg.set_by) {
        applyFaceOverrideUI(true, msg.set_by, msg.image_data || null);
      } else if (msg.success && !msg.set_by) {
        applyFaceOverrideUI(false);
      } else {
        faceStatusEl.textContent = msg.message || "Failed";
        faceStatusEl.className = "face-status error";
      }
      log(`Face swap: ${msg.message}`);
    } else if (msg.type === "face_captured") {
      if (roomState) {
        const peer = roomState.peers.find((p) => p.peer_id === msg.peer_id);
        if (peer) {
          peer.face_captured = msg.success;
          updateRoomUI(roomState);
        }
      }
      updateShuffleBtnState();
      log(`Face captured: ${msg.username} (${msg.peer_id})`);
    } else if (msg.type === "shuffle_applied") {
      shuffleAssignments = msg.assignments || [];
      shuffleStatusEl.textContent = "Faces shuffled!";
      triggerShuffleFlash();
      if (roomState) updateRoomUI(roomState);
      const myAssignment = shuffleAssignments.find((a) => a.peer_id === myPeerId);
      if (myAssignment) {
        log(`Shuffle: you got ${myAssignment.assigned_face_of}'s face`);
      }
      log(`Shuffle applied (${shuffleAssignments.length} peers)`);
    } else if (msg.type === "shuffle_cleared") {
      shuffleAssignments = null;
      shuffleStatusEl.textContent = msg.reason || "";
      if (roomState) updateRoomUI(roomState);
      log(`Shuffle cleared: ${msg.reason || ""}`);
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

// Start camera acquisition early when user interacts with the join form.
usernameInput.addEventListener("focus", _warmCamera, { once: true });
joinOverlay.addEventListener("pointerdown", _warmCamera, { once: true });

startBtn.addEventListener("click", async () => {
  _warmCamera();
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("");
  showJoinLoading("Connecting…", "");
  logEl.textContent = "";
  resetHud();

  if (isLocalMode()) {
    const wsUrl = buildLocalWsUrl();
    log(`Local mode → ${wsUrl}`);
    connectWs(wsUrl);
    return;
  }

  const appId = normalizeAppId(appIdInput.value.trim());
  if (!appId) {
    log("Missing endpoint.");
    clearLoading();
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

  // Lucent path: controller resolves the current pod URL for this app_id.
  // No auth token needed — the pod URL itself is the secret.
  if (ENV_LUCENT_CONTROLLER_URL) {
    updateJoinLoading("Contacting controller…", "");
    try {
      const wsUrl = await resolveLucentPodWsUrl(appId);
      connectWs(wsUrl);
    } catch (err) {
      log(`Failed to resolve pod: ${err.message || err}`);
      clearLoading();
      stop();
      startBtn.disabled = false;
      stopBtn.disabled = true;
      started = false;
    }
    return;
  }

  try {
    if (_prefetchedToken && Date.now() < _prefetchedTokenExpires) {
      authToken = _prefetchedToken;
      _prefetchedToken = null;
      log("Using prefetched token (0ms)");
    } else {
      authToken = await getTemporaryAuthToken(appId);
    }
  } catch (err) {
    log(`Failed to fetch token: ${err.message || err}`);
    clearLoading();
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

// ---- Debug toggle ----

debugToggle.addEventListener("click", () => {
  debugSection.classList.toggle("open");
});

// ---- Shuffle flash ----

const triggerShuffleFlash = () => {
  shuffleFlash.classList.remove("active");
  void shuffleFlash.offsetWidth;
  shuffleFlash.classList.add("active");
  shuffleFlash.addEventListener("animationend", () => {
    shuffleFlash.classList.remove("active");
  }, { once: true });
};
