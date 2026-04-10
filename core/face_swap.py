"""Face swap pipeline: InsightFace detection + inswapper_128 + optional GFPGAN.

Wraps the proven Deep-Live-Cam model stack into a single callable class.
The ``FaceSwapper`` detects all faces in a frame, swaps each with a
pre-registered source face identity, and optionally enhances the result
with GFPGAN face restoration.

Performance notes:
  Tier 1:
  - ``face_app`` (full buffalo_l) is used only for source-face capture and
    ``detect_faces`` where embeddings/landmarks are needed.
  - ``face_app_swap`` (detection-only, 320x320) is used inside
    ``swap_with_source`` — the inswapper only needs bbox + kps from the
    target face and the pre-computed source embedding.
  - The inswapper ONNX session is recreated with ORT_ENABLE_ALL graph
    optimisations and tuned cuDNN workspace settings.
  Tier 2:
  - GPU-side paste_back replaces insightface's CPU ``cv2.warpAffine`` +
    morphological ops with ``torch.nn.functional.grid_sample`` and GPU
    tensor operations, eliminating the CPU bottleneck.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
    """Recreate the ONNX Runtime session on *model* with tuned CUDA settings.

    Applies ORT graph-level optimisations and aggressive cuDNN workspace
    settings.  Does NOT enable CUDA graphs (that requires IOBinding which
    insightface's internal ``session.run()`` calls don't use).
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
        log.info("rebuilt inswapper session with ORT_ENABLE_ALL optimisations")
    except Exception as exc:
        log.warning(f"failed to rebuild inswapper session: {exc}")


def _make_gaussian_kernel(k: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel for conv2d blurring."""
    blur_size = 2 * k + 1
    ax = torch.arange(blur_size, dtype=torch.float32, device=device) - k
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    # sigma=0 convention in OpenCV means sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, blur_size, blur_size)


def _gpu_paste_back(
    target_img: np.ndarray,
    bgr_fake: np.ndarray,
    aimg: np.ndarray,
    M: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """GPU-accelerated paste_back — replaces insightface's CPU path.

    Replicates the blend logic from ``INSwapper.get(paste_back=True)``
    (inswapper.py lines 59-104) but runs warpAffine, morphological ops,
    and blending on GPU via PyTorch.

    Note: the original code computes a ``fake_diff`` mask but never uses
    it in the final blend (the assignment ``img_mask = fake_diff`` on
    line 100 is commented out).  We skip it entirely.
    """
    H, W = target_img.shape[:2]
    crop_h, crop_w = aimg.shape[:2]

    # ── 1. Inverse affine matrix (2x3, trivial on CPU) ────────────────
    IM = cv2.invertAffineTransform(M)

    # ── 2. Build normalised theta for affine_grid ─────────────────────
    # IM maps target_px → crop_px.  We convert to normalised [-1,1] coords
    # that torch.nn.functional.affine_grid / grid_sample expect.
    A = IM[:, :2]  # 2x2 rotation/scale
    t = IM[:, 2]   # translation

    sx_out = (W - 1) / 2.0
    sy_out = (H - 1) / 2.0
    sx_in = (crop_w - 1) / 2.0
    sy_in = (crop_h - 1) / 2.0

    S_in = np.array([[1.0 / sx_in, 0], [0, 1.0 / sy_in]])
    S_out = np.array([[sx_out, 0], [0, sy_out]])

    A_norm = S_in @ A @ S_out
    t_norm = S_in @ (A @ np.array([sx_out, sy_out]) + t) - np.array([1.0, 1.0])

    theta = np.zeros((1, 2, 3), dtype=np.float32)
    theta[0, :2, :2] = A_norm
    theta[0, :2, 2] = t_norm

    theta_t = torch.from_numpy(theta).to(device)
    grid = F.affine_grid(theta_t, (1, 1, H, W), align_corners=True)

    # ── 3. Warp bgr_fake (3ch) + white mask (1ch) in one grid_sample ─
    img_white = np.full((crop_h, crop_w), 255.0, dtype=np.float32)

    bgr_t = torch.from_numpy(bgr_fake.astype(np.float32)).to(device)      # [h, w, 3]
    white_t = torch.from_numpy(img_white).to(device).unsqueeze(-1)          # [h, w, 1]

    stacked = torch.cat([bgr_t, white_t], dim=-1).unsqueeze(0).permute(0, 3, 1, 2)  # [1, 4, h, w]
    warped = F.grid_sample(stacked, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    bgr_warped = warped[0, :3]     # [3, H, W]
    white_warped = warped[0, 3:4]  # [1, H, W]

    # ── 4. Threshold white mask ───────────────────────────────────────
    img_mask = torch.where(white_warped > 20, 255.0, 0.0)

    # ── 5. Compute dynamic kernel size from mask bounds ───────────────
    mask_yx = torch.where(img_mask[0] == 255)
    if len(mask_yx[0]) == 0:
        return target_img
    mask_h = (mask_yx[0].max() - mask_yx[0].min()).item()
    mask_w = (mask_yx[1].max() - mask_yx[1].min()).item()
    mask_size = int(np.sqrt(mask_h * mask_w))

    # ── 6. Erode img_mask (min-pool = erode for white-on-black) ───────
    k_erode = max(mask_size // 10, 10)
    if k_erode % 2 == 0:
        k_erode += 1  # odd kernel keeps spatial dims with padding=k//2
    pad_e = k_erode // 2
    img_mask = -F.max_pool2d(-img_mask.unsqueeze(0), kernel_size=k_erode, stride=1, padding=pad_e)[0]

    # ── 7. Gaussian blur ──────────────────────────────────────────────
    k_blur = max(mask_size // 20, 5)
    g = _make_gaussian_kernel(k_blur, device)
    img_mask = F.conv2d(img_mask.unsqueeze(0), g, padding=k_blur)[0]

    # ── 8. Normalise to [0, 1] and blend ──────────────────────────────
    img_mask = img_mask / 255.0  # [1, H, W]
    target_t = torch.from_numpy(target_img.astype(np.float32)).to(device).permute(2, 0, 1)
    result = img_mask * bgr_warped + (1.0 - img_mask) * target_t

    return result.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()


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
            self._torch_device = torch.device("cuda")
        else:
            self._torch_device = torch.device("cpu")

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

        When running on CUDA the CPU-heavy ``paste_back`` inside insightface
        is replaced with a GPU-accelerated path (``_gpu_paste_back``).

        Set *copy=False* when the caller does not need *frame_bgr* preserved
        (e.g. the room processing loop) to avoid an allocation per frame.
        """
        from insightface.utils import face_align

        faces = self.face_app_swap.get(frame_bgr)
        if not faces:
            return frame_bgr
        result = frame_bgr.copy() if copy else frame_bgr
        use_gpu = self._device == "cuda"
        for face in faces:
            if use_gpu:
                aimg, M = face_align.norm_crop2(result, face.kps, self.swapper.input_size[0])
                bgr_fake, _ = self.swapper.get(result, face, source_face, paste_back=False)
                result = _gpu_paste_back(result, bgr_fake, aimg, M, self._torch_device)
            else:
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
        """Run a dummy frame through both analysers to warm up ONNX sessions."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.face_app.get(dummy)
        _ = self.face_app_swap.get(dummy)
        log.info("warmup complete")
