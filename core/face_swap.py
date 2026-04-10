"""Face swap pipeline: InsightFace detection + inswapper_128 + optional GFPGAN.

Wraps the proven Deep-Live-Cam model stack into a single callable class.
The ``FaceSwapper`` detects all faces in a frame, swaps each with a
pre-registered source face identity, and optionally enhances the result
with GFPGAN face restoration.
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


class FaceSwapper:
    """Full face-swap pipeline: detect -> swap -> (optional) enhance."""

    def __init__(self, device: str = "cpu") -> None:
        import insightface
        from insightface.app import FaceAnalysis

        self._device = device
        ctx_id = 0 if device == "cuda" else -1

        log.info(f"loading buffalo_l face analyser (ctx_id={ctx_id})")
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        model_path = _download_inswapper()
        log.info(f"loading inswapper from {model_path}")
        self.swapper = insightface.model_zoo.get_model(model_path)

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

    def swap_with_source(self, frame_bgr: np.ndarray, source_face: Any) -> np.ndarray:
        """Swap all detected faces in frame to the given source face identity."""
        faces = self.face_app.get(frame_bgr)
        if not faces:
            return frame_bgr
        result = frame_bgr.copy()
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
        """Run a dummy frame through detection to warm up ONNX sessions."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.face_app.get(dummy)
        log.info("warmup complete")
