"""Frame composition: draw YOLO bounding boxes onto a BGR frame."""

from __future__ import annotations

import hashlib
from typing import Optional

from .yolo import YoloBox


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


def compose_frame(frame_bgr, yolo: Optional[list[YoloBox]] = None):
    """Draw YOLO boxes onto a copy of the frame. Returns a uint8 BGR ndarray."""
    out = frame_bgr.copy()
    if yolo:
        out = _draw_boxes(out, yolo)
    return out
