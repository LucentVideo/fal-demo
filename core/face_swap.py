"""Face swap pipeline: InsightFace detection + inswapper_128 + optional GFPGAN.

Wraps the proven Deep-Live-Cam model stack into a single callable class.
The ``FaceSwapper`` detects all faces in a frame, swaps each with a
pre-registered source face identity, and optionally enhances the result
with GFPGAN face restoration.

Performance notes:
  - ``face_app`` (full buffalo_l) is used only for source-face capture and
    ``detect_faces`` where embeddings/landmarks are needed.
  - ``face_app_swap`` (detection-only, 320x320) is used inside
    ``swap_with_source`` — the inswapper only needs bbox + kps from the
    target face and the pre-computed source embedding.
  - The inswapper ONNX session is recreated with ORT_ENABLE_ALL graph
    optimisations and tuned cuDNN workspace settings.
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


_GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
_GFPGAN_MIN_BYTES = 300_000_000  # ~348 MB expected


def _download_gfpgan() -> str:
    """Download GFPGANv1.4.pth weights, returning the local path."""
    import os
    import urllib.request

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gfpgan")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "GFPGANv1.4.pth")

    if os.path.exists(model_path) and os.path.getsize(model_path) < _GFPGAN_MIN_BYTES:
        log.warning("removing incomplete GFPGANv1.4 download")
        os.remove(model_path)

    if not os.path.exists(model_path):
        log.info(f"downloading GFPGANv1.4 weights to {model_path} ...")
        urllib.request.urlretrieve(_GFPGAN_URL, model_path)
        log.info("download complete")

    return model_path


def _rebuild_session_optimised(model) -> None:
    """Recreate the ONNX Runtime session with the tuned CUDA EP path."""
    try:
        import onnxruntime as ort

        model_path = getattr(model, "model_file", None)
        if model_path is None:
            log.warning("cannot rebuild session: model_file not found")
            return

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_opts = {
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv_use_max_workspace": "1",
            "arena_extend_strategy": "kSameAsRequested",
        }

        providers: list = [
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]

        model.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers,
        )

        active = model.session.get_providers()
        log.info(f"rebuilt inswapper session — active providers: {active}")
    except Exception as exc:
        log.warning(f"failed to rebuild inswapper session: {exc}")


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
            _rebuild_session_optimised(self.swapper)

        # Download GFPGAN weights and load the enhancer eagerly so startup
        # fails fast rather than on the first enhanced frame.
        import sys
        import torchvision.transforms.functional as _F
        sys.modules.setdefault("torchvision.transforms.functional_tensor", _F)

        self._enhancer: Any = None
        try:
            from gfpgan import GFPGANer

            gfpgan_path = _download_gfpgan()
            self._enhancer = GFPGANer(
                model_path=gfpgan_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
            log.info("GFPGAN enhancer loaded")
        except Exception as exc:
            log.warning(f"GFPGAN unavailable (non-fatal): {exc}")

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

    def _enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
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
        """Run a dummy frame through analysers and inswapper to warm up sessions."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.face_app.get(dummy)
        _ = self.face_app_swap.get(dummy)

        try:
            sess = self.swapper.session
            input_names = [i.name for i in sess.get_inputs()]
            input_shapes = [i.shape for i in sess.get_inputs()]
            log.info(f"inswapper warmup: inputs={list(zip(input_names, input_shapes))}")
            dummy_inputs = {
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in zip(input_names, input_shapes)
            }
            _ = sess.run(None, dummy_inputs)
            log.info("inswapper warmup complete")
        except Exception as exc:
            log.warning(f"inswapper warmup failed (non-fatal): {exc}")

        log.info("warmup complete")
