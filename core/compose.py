"""Frame composition: turn the perceptual outputs into a BGR frame to
hand back to the viewer. Four modes, selected by ``layer``:

* ``"detection"``   — raw frame + YOLO bounding boxes
* ``"depth"``       — depth map colormapped (turbo)
* ``"segmentation"``— raw frame with semantic masks alpha-blended
* ``"composite"``   — all three layered: segmentation alpha-blend, YOLO
                      boxes on top, depth picture-in-picture bottom-right
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .perception import YoloBox, SegRegion


def _class_color(name: str) -> tuple[int, int, int]:
    """Deterministic BGR color from a class label."""
    h = hashlib.md5(name.encode("utf-8")).digest()
    return int(h[0]), int(h[1]), int(h[2])


def _draw_boxes(img, boxes: list[YoloBox]):
    import cv2

    for box in boxes:
        color = _class_color(box.label)
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{box.label} {box.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img,
            (p1[0], max(0, p1[1] - th - 6)),
            (p1[0] + tw + 4, p1[1]),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (p1[0] + 2, max(th, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def _depth_to_bgr(depth):
    """Normalized float [0,1] HxW -> turbo-colormapped BGR uint8 HxWx3."""
    import cv2
    import numpy as np

    d8 = (depth * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def _blend_segmentation(img, regions: list[SegRegion], alpha: float = 0.35):
    import numpy as np

    overlay = img.copy()
    for region in regions:
        color = _class_color(region.label)
        overlay[region.mask] = color
    return (img.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(
        img.dtype
    )


def _draw_seg_labels(img, regions: list[SegRegion]):
    import cv2
    import numpy as np

    seen = set()
    y = 18
    for region in regions:
        if region.label in seen:
            continue
        seen.add(region.label)
        color = _class_color(region.label)
        cv2.putText(
            img,
            region.label,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 16
        if y > img.shape[0] - 8:
            break
    return img


def _pip_depth(img, depth):
    """Paste a depth thumbnail into the bottom-right of the frame."""
    import cv2

    h, w = img.shape[:2]
    pip_w = max(120, w // 4)
    pip_h = int(pip_w * depth.shape[0] / depth.shape[1])
    thumb = cv2.resize(depth, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
    thumb_bgr = _depth_to_bgr(thumb)
    x0 = w - pip_w - 8
    y0 = h - pip_h - 8
    img[y0 : y0 + pip_h, x0 : x0 + pip_w] = thumb_bgr
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pip_w, y0 + pip_h), (255, 255, 255), 1)
    return img


def compose_frame(
    frame_bgr,
    layer: str,
    yolo: Optional[list[YoloBox]] = None,
    depth=None,
    seg: Optional[list[SegRegion]] = None,
):
    """Render the active layer. Mutates/returns a uint8 BGR numpy array."""
    if layer == "depth":
        if depth is None:
            return frame_bgr
        return _depth_to_bgr(depth)

    if layer == "detection":
        out = frame_bgr.copy()
        if yolo:
            out = _draw_boxes(out, yolo)
        return out

    if layer == "segmentation":
        out = frame_bgr.copy()
        if seg:
            out = _blend_segmentation(out, seg, alpha=0.45)
            out = _draw_seg_labels(out, seg)
        return out

    # composite
    out = frame_bgr.copy()
    if seg:
        out = _blend_segmentation(out, seg, alpha=0.30)
    if yolo:
        out = _draw_boxes(out, yolo)
    if depth is not None:
        out = _pip_depth(out, depth)
    return out
