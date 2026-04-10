"""Face swap pipeline: InsightFace detection + inswapper_128 + optional GFPGAN.

Wraps the proven Deep-Live-Cam model stack into a single callable class.
The ``FaceSwapper`` detects all faces in a frame, swaps each with a
pre-registered source face identity, and optionally enhances the result
with GFPGAN face restoration.

Performance notes (Tier 1 optimisations):
  - ``face_app`` (full buffalo_l) is used only for source-face capture and
    ``detect_faces`` where embeddings/landmarks are needed.
  - ``face_app_swap`` (detection-only, 320x320) is used inside
    ``swap_with_source`` — the inswapper only needs bbox + kps from the
    target face and the pre-computed source embedding.
  - The inswapper ONNX session is recreated with ``enable_cuda_graph=True``
    to eliminate kernel-launch overhead on the fixed 128x128 input.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("face_swap")
log.setLevel(logging.DEBUG)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    log.addHandler(_h)


def _download_inswapper() -> str:
    """Download inswapper_128_fp16.onnx from HuggingFace (Deep-Live-Cam weights)."""
    import os
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="hacksider/deep-live-cam",
        filename="inswapper_128_fp16.onnx",
        token=os.environ.get("HF_TOKEN"),
    )


def _rebuild_session_with_cuda_graph(model) -> None:
    """Recreate the ONNX Runtime session on *model* with CUDA graphs enabled.

    CUDA graphs capture the full GPU kernel sequence on the first run and
    replay it on subsequent runs, eliminating per-frame launch overhead.
    Only safe for fixed-shape inputs (inswapper is always [1,3,128,128]).
    """
    try:
        import onnxruntime as ort

        model_path = getattr(model, "model_file", None)
        if model_path is None:
            log.warning("cannot rebuild session: model_file not found")
            return

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "enable_cuda_graph": "1",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "arena_extend_strategy": "kSameAsRequested",
                },
            ),
            "CPUExecutionProvider",
        ]
        model.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers,
        )
        log.info(f"rebuilt inswapper session with CUDA graphs enabled")
    except Exception as exc:
        log.warning(f"failed to enable CUDA graphs on inswapper: {exc}")


class FaceSwapper:
    """Full face-swap pipeline: detect -> swap -> (optional) enhance."""

    def __init__(self, device: str = "cpu") -> None:
        import insightface
        from insightface.app import FaceAnalysis

        self._device = device
        ctx_id = 0 if device == "cuda" else -1

        # Full analyser — used for source-face capture (needs recognition
        # embedding) and detect_faces() where callers need all attributes.
        log.info(f"loading buffalo_l face analyser (ctx_id={ctx_id})")
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Lightweight detection-only analyser for the swap hot-path.
        # Skips landmark_3d_68, landmark_2d_106, gender/age, and recognition
        # models that the inswapper never reads from the *target* face.
        # Uses 320x320 det_size — sufficient for typical webcam feeds.
        log.info("loading detection-only analyser for swap path")
        self.face_app_swap = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"],
        )
        self.face_app_swap.prepare(ctx_id=ctx_id, det_size=(320, 320))

        model_path = _download_inswapper()
        log.info(f"loading inswapper from {model_path}")
        self.swapper = insightface.model_zoo.get_model(model_path)

        if device == "cuda":
            _rebuild_session_with_cuda_graph(self.swapper)

        self._enhancer: Any = None
        self.enhance_enabled = False
        self.source_face: Any = None

        log.info("FaceSwapper ready")

    def set_source(self, img_bgr: np.ndarray) -> bool:
        """Analyse source image, store the largest detected face. Returns True on success."""
        faces = self.face_app.get(img_bgr)
        if not faces:
            log.info("set_source: no faces detected in image")
            return False
        self.source_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        log.info("set_source: source face set")
        return True

    def detect_faces(self, img_bgr: np.ndarray) -> list:
        """Run face detection + embedding extraction without requiring a source face."""
        return self.face_app.get(img_bgr)

    def clear_source(self) -> None:
        self.source_face = None
        log.info("source face cleared")

    def swap_with_source(
        self, frame_bgr: np.ndarray, source_face: Any, *, copy: bool = True,
    ) -> np.ndarray:
        """Swap all detected faces in frame to the given source face identity.

        Uses the lightweight detection-only analyser (no recognition /
        landmarks / gender-age) since the inswapper only reads bbox + kps
        from the target face.

        Set *copy=False* when the caller does not need *frame_bgr* preserved
        (e.g. the room processing loop) to avoid an allocation per frame.
        """
        faces = self.face_app_swap.get(frame_bgr)
        if not faces:
            return frame_bgr
        result = frame_bgr.copy() if copy else frame_bgr
        for face in faces:
            result = self.swapper.get(result, face, source_face, paste_back=True)
        if self.enhance_enabled:
            result = self._enhance(result)
        return result

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect all faces in frame, swap each with the source identity."""
        if self.source_face is None:
            return frame_bgr
        return self.swap_with_source(frame_bgr, self.source_face)

    def _ensure_enhancer(self) -> None:
        if self._enhancer is not None:
            return
        try:
            from gfpgan import GFPGANer

            self._enhancer = GFPGANer(
                model_path="GFPGANv1.4.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
            log.info("GFPGAN enhancer loaded")
        except Exception as exc:
            log.warning(f"failed to load GFPGAN: {exc}")
            self._enhancer = None

    def _enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._ensure_enhancer()
        if self._enhancer is None:
            return frame_bgr
        try:
            _, _, output = self._enhancer.enhance(
                frame_bgr, has_aligned=False, only_center_face=False, paste_back=True,
            )
            return output
        except Exception:
            return frame_bgr

    def warmup(self) -> None:
        """Run a dummy frame through both analysers to warm up ONNX sessions."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.face_app.get(dummy)
        _ = self.face_app_swap.get(dummy)
        log.info("warmup complete")
