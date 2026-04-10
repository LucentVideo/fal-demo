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


def _build_gpu_paste_back():
    """Build a GPU replacement for ``INSwapper.get(paste_back=True)``.

    The upstream method (``insightface/model_zoo/inswapper.py``) runs the ONNX
    swap on GPU but does **all** of the post-processing — three full-frame
    affine warps, an erode, a dilate, two Gaussian blurs, and the final alpha
    blend — on CPU via numpy/OpenCV.  At 640×480 with one face per frame
    that's a few ms of CPU work *and* a forced GPU→CPU sync on the swap
    output every frame.

    This factory returns a drop-in ``get`` method that keeps the post-process
    on the GPU using ``cupy`` + ``cupyx.scipy.ndimage``.  Returns ``None`` if
    cupy isn't importable, so the caller can fall back to the CPU path.

    Notes on correctness vs. the upstream code:
      - ``cv2.warpAffine(src, IM, dsize)`` (no WARP_INVERSE_MAP) inverts ``IM``
        internally, so the *effective* inverse map is the original ``M``.
        ``cupyx.scipy.ndimage.affine_transform`` already takes an inverse map,
        so we pass ``M`` directly (not ``IM``).
      - cv2's 2×3 ``M`` is in ``(x, y)`` order; ndimage uses ``(y, x)``.  See
        the matrix swap below.
      - ``cv2.GaussianBlur(ksize=k, sigma=0)`` derives
        ``sigma = 0.3*((k-1)*0.5 - 1) + 0.8``.  We compute the same sigma so
        the blur strength matches the upstream behaviour.
      - The upstream ``fake_diff`` mask is computed and then *never used* in
        the final blend (the line that would consume it is commented out).
        We skip computing it entirely.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import (
            affine_transform,
            gaussian_filter,
            grey_erosion,
        )
    except Exception as exc:
        log.warning(f"cupy unavailable, GPU paste_back disabled: {exc}")
        return None

    import cv2
    from insightface.utils import face_align

    def _gpu_get(self, img, target_face, source_face, paste_back=True):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(
            self.output_names,
            {self.input_names[0]: blob, self.input_names[1]: latent},
        )[0]

        # Move pred to GPU as float32 BGR (the upstream model emits RGB; flip
        # the channel axis with fancy indexing so the result is contiguous).
        pred_gpu = cp.asarray(pred, dtype=cp.float32)  # (1, 3, h, w)
        bgr_fake_gpu = cp.clip(
            pred_gpu[0, ::-1].transpose(1, 2, 0) * 255.0, 0.0, 255.0,
        )
        bgr_fake_gpu = cp.ascontiguousarray(bgr_fake_gpu)

        if not paste_back:
            bgr_fake_np = cp.asnumpy(bgr_fake_gpu).astype(np.uint8)
            return bgr_fake_np, M

        target_h, target_w = img.shape[:2]
        aimg_h, aimg_w = aimg.shape[:2]

        # Upload target frame as uint8 (1× bandwidth) and cast on device.
        target_gpu = cp.asarray(img).astype(cp.float32)

        # cv2 forward 2×3 M maps target → crop in (x, y) order.  ndimage's
        # affine_transform takes an inverse map in (y, x) order, so:
        matrix2 = cp.asarray(
            [[M[1, 1], M[1, 0]], [M[0, 1], M[0, 0]]],
            dtype=cp.float32,
        )
        offset2 = cp.asarray([M[1, 2], M[0, 2]], dtype=cp.float32)

        # Warp the swap output back into the target frame, per channel so we
        # don't pay for spurious 3-D interpolation along the channel axis.
        bgr_fake_warped = cp.empty((target_h, target_w, 3), dtype=cp.float32)
        for c in range(3):
            bgr_fake_warped[:, :, c] = affine_transform(
                bgr_fake_gpu[:, :, c],
                matrix2,
                offset=offset2,
                output_shape=(target_h, target_w),
                order=1,
                mode="constant",
                cval=0.0,
            )

        # Build the alpha mask by warping a constant-255 plate the same way.
        img_white_src = cp.full((aimg_h, aimg_w), 255.0, dtype=cp.float32)
        img_mask = affine_transform(
            img_white_src,
            matrix2,
            offset=offset2,
            output_shape=(target_h, target_w),
            order=1,
            mode="constant",
            cval=0.0,
        )
        img_mask = cp.where(img_mask > 20.0, 255.0, img_mask)

        # mask_size estimate (matches upstream): bbox of pixels at 255 in the
        # warped, thresholded mask.  One small host sync per face.
        mask_inds_y, mask_inds_x = cp.where(img_mask == 255.0)
        if mask_inds_y.size == 0:
            return img.copy()
        mh = int((mask_inds_y.max() - mask_inds_y.min()).get())
        mw = int((mask_inds_x.max() - mask_inds_x.min()).get())
        mask_size = int(np.sqrt(mh * mw))

        k_erode = max(mask_size // 10, 10)
        img_mask = grey_erosion(img_mask, size=(k_erode, k_erode))

        # cv2.GaussianBlur(k, sigma=0) → sigma = 0.3*((k-1)/2 - 1) + 0.8.
        # Upstream's k is (2*kk+1), so sigma simplifies to 0.3*(kk-1) + 0.8.
        kk1 = max(mask_size // 20, 5)
        sigma1 = 0.3 * (kk1 - 1) + 0.8
        img_mask = gaussian_filter(img_mask, sigma=sigma1)

        img_mask = (img_mask / 255.0)[:, :, None]

        fake_merged = img_mask * bgr_fake_warped + (1.0 - img_mask) * target_gpu
        fake_merged = cp.clip(fake_merged, 0.0, 255.0).astype(cp.uint8)
        return cp.asnumpy(fake_merged)

    return _gpu_get


_GPU_PASTE_BACK_GET = None


def _gpu_paste_back_enabled() -> bool:
    """Toggle for the cupy paste_back path.

    Set ``FAL_GPU_PASTE_BACK=0`` (or ``false`` / ``no`` / ``off``) to force
    the original insightface CPU code path.  Default is enabled — this is
    the whole point of the perf branch — but the kill-switch is here so any
    visual regression in production can be reverted without a redeploy.
    """
    import os

    raw = os.environ.get("FAL_GPU_PASTE_BACK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _patch_swapper_paste_back_gpu(swapper) -> None:
    """Replace ``swapper.get`` with the cupy-backed version, if available
    and not disabled via ``FAL_GPU_PASTE_BACK=0``."""
    if not _gpu_paste_back_enabled():
        log.info("inswapper paste_back forced to CPU (FAL_GPU_PASTE_BACK=0)")
        return

    global _GPU_PASTE_BACK_GET
    if _GPU_PASTE_BACK_GET is None:
        _GPU_PASTE_BACK_GET = _build_gpu_paste_back()
    if _GPU_PASTE_BACK_GET is None:
        log.info("inswapper paste_back staying on CPU (no cupy)")
        return
    import types

    swapper.get = types.MethodType(_GPU_PASTE_BACK_GET, swapper)
    log.info("inswapper.get patched: paste_back now runs on GPU via cupy")


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
            _patch_swapper_paste_back_gpu(self.swapper)

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
