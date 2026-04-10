"""Verify the cupy GPU paste_back rewrite is mathematically equivalent to
the upstream insightface CPU paste_back, without needing a GPU.

We use ``scipy.ndimage.affine_transform`` as a stand-in for
``cupyx.scipy.ndimage.affine_transform`` — they share the same API and
semantics, so if the matrix conversion + ndimage call match cv2 here, they
will also match on the GPU.

Run:
    .venv/bin/python scripts/verify_gpu_paste_back_math.py

The script reports max-abs and mean-abs pixel diff for each stage.  Linear
interpolation rounding gives ~0.5–1 LSB jitter; anything above ~2 LSB
indicates the matrix conversion is wrong.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import (
    affine_transform,
    gaussian_filter,
    grey_erosion,
)


# ---------------------------------------------------------------------------
# Reference CPU implementation — copied verbatim from
# insightface/model_zoo/inswapper.py (paste_back branch), minus the dead
# fake_diff path.
# ---------------------------------------------------------------------------

def cpu_paste_back(target_img: np.ndarray, bgr_fake: np.ndarray, M: np.ndarray) -> np.ndarray:
    aimg_h, aimg_w = bgr_fake.shape[:2]
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((aimg_h, aimg_w), 255, dtype=np.float32)
    bgr_fake_w = cv2.warpAffine(
        bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0,
    )
    img_white_w = cv2.warpAffine(
        img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0,
    )
    img_white_w[img_white_w > 20] = 255
    img_mask = img_white_w
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if mask_h_inds.size == 0:
        return target_img.copy()
    mask_h = mask_h_inds.max() - mask_h_inds.min()
    mask_w = mask_w_inds.max() - mask_w_inds.min()
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    blur = (2 * k + 1, 2 * k + 1)
    img_mask = cv2.GaussianBlur(img_mask, blur, 0)
    img_mask = (img_mask / 255.0)[:, :, None]
    out = img_mask * bgr_fake_w + (1 - img_mask) * target_img.astype(np.float32)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Mirror of the cupy GPU implementation, but using scipy.ndimage on the CPU.
# This is the code under test — if scipy here matches cv2, the cupy version
# (same API) will too.
# ---------------------------------------------------------------------------

def gpu_paste_back_cpu_twin(target_img: np.ndarray, bgr_fake: np.ndarray, M: np.ndarray) -> np.ndarray:
    target_h, target_w = target_img.shape[:2]
    aimg_h, aimg_w = bgr_fake.shape[:2]

    target_f = target_img.astype(np.float32)
    bgr_fake_f = bgr_fake.astype(np.float32)

    # cv2 forward (x, y) → ndimage inverse (y, x)
    matrix2 = np.asarray(
        [[M[1, 1], M[1, 0]], [M[0, 1], M[0, 0]]], dtype=np.float32,
    )
    offset2 = np.asarray([M[1, 2], M[0, 2]], dtype=np.float32)

    bgr_fake_warped = np.empty((target_h, target_w, 3), dtype=np.float32)
    for c in range(3):
        bgr_fake_warped[:, :, c] = affine_transform(
            bgr_fake_f[:, :, c],
            matrix2,
            offset=offset2,
            output_shape=(target_h, target_w),
            order=1,
            mode="constant",
            cval=0.0,
        )

    img_white = np.full((aimg_h, aimg_w), 255.0, dtype=np.float32)
    img_mask = affine_transform(
        img_white,
        matrix2,
        offset=offset2,
        output_shape=(target_h, target_w),
        order=1,
        mode="constant",
        cval=0.0,
    )
    img_mask = np.where(img_mask > 20.0, 255.0, img_mask)

    inds_y, inds_x = np.where(img_mask == 255.0)
    if inds_y.size == 0:
        return target_img.copy()
    mh = int(inds_y.max() - inds_y.min())
    mw = int(inds_x.max() - inds_x.min())
    mask_size = int(np.sqrt(mh * mw))

    k_erode = max(mask_size // 10, 10)
    img_mask = grey_erosion(img_mask, size=(k_erode, k_erode))

    kk1 = max(mask_size // 20, 5)
    sigma1 = 0.3 * (kk1 - 1) + 0.8
    img_mask = gaussian_filter(img_mask, sigma=sigma1)

    img_mask = (img_mask / 255.0)[:, :, None]
    merged = img_mask * bgr_fake_warped + (1.0 - img_mask) * target_f
    return np.clip(merged, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_similarity_M(image_size: int = 128, target_h: int = 480, target_w: int = 640) -> np.ndarray:
    """Build a similarity transform like ``estimate_norm`` produces — random
    rotation, uniform scale, translation that places a face-sized region
    inside the target frame."""
    rng = np.random.default_rng(42)
    angle = float(rng.uniform(-25, 25))     # degrees
    scale = float(rng.uniform(1.5, 3.5))    # how big the face is in target
    tx = float(rng.uniform(50, target_w - 200))
    ty = float(rng.uniform(50, target_h - 200))

    # Compose: target → crop is (rotate + scale around centre) → translate
    centre_target = np.array([tx, ty], dtype=np.float32)
    centre_crop = np.array([image_size / 2, image_size / 2], dtype=np.float32)
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float32) * scale
    t = centre_crop - R @ centre_target
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = R
    M[:2, 2] = t
    return M


def make_target_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Synthetic target frame with a recognisable gradient + checker so any
    misalignment is obvious."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[..., 0] = (xx / w * 255)
    img[..., 1] = (yy / h * 255)
    img[..., 2] = ((xx + yy) % 64) * 4
    return img.astype(np.uint8)


def make_fake_face(image_size: int = 128) -> np.ndarray:
    """Synthetic 'swap output' — concentric rings + colour bias so any wrong
    rotation/scale/sign in the warp shows up immediately."""
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    cy = cx = image_size / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    img[..., 0] = (np.sin(r * 0.4) * 0.5 + 0.5) * 255   # B
    img[..., 1] = (xx / image_size) * 255                # G
    img[..., 2] = (yy / image_size) * 255                # R
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    total = diff.size
    n_diff = int((diff > 0).sum())
    n_big = int((diff > 5).sum())
    print(
        f"  {name:30s} max={diff.max():4d}  mean={diff.mean():.4f}  "
        f">0={100*n_diff/total:5.2f}%  >5={100*n_big/total:6.4f}%  "
        f"shape={a.shape}"
    )


def test_warp_only() -> None:
    """Stage 1: just the warp, in isolation, no masking/blur."""
    print("[stage 1] warp-only equivalence")
    target = make_target_image()
    fake = make_fake_face()
    M = make_similarity_M()

    # cv2 reference
    cv2_warped = cv2.warpAffine(
        fake, cv2.invertAffineTransform(M), (target.shape[1], target.shape[0]),
        borderValue=0.0,
    )

    # ndimage twin
    matrix2 = np.asarray(
        [[M[1, 1], M[1, 0]], [M[0, 1], M[0, 0]]], dtype=np.float32,
    )
    offset2 = np.asarray([M[1, 2], M[0, 2]], dtype=np.float32)
    nd_warped = np.empty_like(cv2_warped, dtype=np.float32)
    for c in range(3):
        nd_warped[:, :, c] = affine_transform(
            fake[:, :, c].astype(np.float32),
            matrix2,
            offset=offset2,
            output_shape=cv2_warped.shape[:2],
            order=1,
            mode="constant",
            cval=0.0,
        )
    nd_warped_u8 = np.clip(nd_warped, 0, 255).astype(np.uint8)

    report("warp(fake) bgr", cv2_warped, nd_warped_u8)

    # Localise where the diffs live: are they on the edge of the warped
    # region (boundary interp difference, harmless) or scattered through the
    # interior (matrix wrong, fatal)?
    diff = np.abs(cv2_warped.astype(np.int32) - nd_warped_u8.astype(np.int32)).max(axis=2)
    big = diff > 5
    if big.any():
        # Mask of where the warp landed (any non-zero pixel in cv2_warped)
        landed = (cv2_warped > 0).any(axis=2).astype(np.uint8)
        # Erode it; pixels in the warp region but not in the eroded region
        # are the boundary.
        eroded = cv2.erode(landed, np.ones((3, 3), np.uint8))
        boundary = landed.astype(bool) & ~eroded.astype(bool)
        on_boundary = int((big & boundary).sum())
        big_total = int(big.sum())
        print(
            f"  -> diffs >5: {big_total} pixels, "
            f"{100*on_boundary/big_total:5.1f}% on the warp boundary"
        )


def test_full_pipeline() -> None:
    """Stage 2: full upstream paste_back vs the cupy-twin pipeline."""
    print("[stage 2] full paste_back equivalence")
    target = make_target_image()
    fake = make_fake_face()
    M = make_similarity_M()

    cpu_out = cpu_paste_back(target, fake, M)
    twin_out = gpu_paste_back_cpu_twin(target, fake, M)
    report("paste_back full", cpu_out, twin_out)


def test_many_seeds() -> None:
    """Stage 3: sweep many random affine setups so we don't fluke one match."""
    print("[stage 3] sweep over 8 random affines")
    target = make_target_image()
    fake = make_fake_face()
    rng = np.random.default_rng(0)
    worst = 0
    for i in range(8):
        # different seed each iter
        angle = float(rng.uniform(-45, 45))
        scale = float(rng.uniform(1.2, 4.0))
        tx = float(rng.uniform(80, 560))
        ty = float(rng.uniform(80, 400))
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32) * scale
        centre_target = np.array([tx, ty], dtype=np.float32)
        centre_crop = np.array([64, 64], dtype=np.float32)
        t = centre_crop - R @ centre_target
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = R
        M[:2, 2] = t

        cpu_out = cpu_paste_back(target, fake, M)
        twin_out = gpu_paste_back_cpu_twin(target, fake, M)
        d = np.abs(cpu_out.astype(np.int32) - twin_out.astype(np.int32))
        print(f"  seed {i}: angle={angle:+.1f}  scale={scale:.2f}  "
              f"max={d.max():4d}  mean={d.mean():.3f}")
        worst = max(worst, int(d.max()))
    print(f"  worst max diff across all seeds: {worst}")


if __name__ == "__main__":
    test_warp_only()
    print()
    test_full_pipeline()
    print()
    test_many_seeds()
