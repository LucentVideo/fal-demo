"""Ultralytics YOLOv8n object detector wrapper.

Takes a numpy BGR uint8 frame and returns a list of ``YoloBox``. Heavy
imports (torch, ultralytics) live inside ``__init__`` so this module is
cheap to import on the fal runner's spec pass and in local smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class YoloBox:
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    label: str
    score: float


class YoloDetector:
    """Ultralytics YOLOv8n object detector.

    Mirrors the load pattern of upstream ``yolo_webcam_webrtc/yolo.py``: the
    weights live at ``/data/yolov8n.pt`` on the fal runner's persistent
    volume. Override with ``weights_path`` for local smoke tests.
    """

    def __init__(self, weights_path: str = "yolov8n.pt", device: str | None = None):
        from ultralytics import YOLO

        from core import detect_device

        self.model = YOLO(weights_path)
        self.device = device or detect_device()

    def __call__(self, frame_bgr) -> list[YoloBox]:
        result = self.model(
            frame_bgr, verbose=False, device=self.device,
            half=(self.device == "cuda"),
        )[0]
        boxes: list[YoloBox] = []
        names = result.names
        if result.boxes is None:
            return boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            boxes.append(
                YoloBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    cls_id=cls_id,
                    label=names.get(cls_id, "obj") if isinstance(names, dict) else str(cls_id),
                    score=score,
                )
            )
        return boxes
