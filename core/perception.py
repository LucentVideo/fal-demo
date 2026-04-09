"""Three perceptual model wrappers, one class each. No fal imports.

Each wrapper takes a numpy BGR uint8 frame and returns a small, uniform
python-native output that ``compose.py`` knows how to render. Heavy imports
(torch, transformers, ultralytics) live inside __init__ so this module is
cheap to import on the fal runner's spec pass and in local smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------- YOLO ---------------------------------------------------------

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

    def __init__(self, weights_path: str = "/data/yolov8n.pt", device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        # ultralytics honors device on the call site, not on load.
        self.device = device

    def __call__(self, frame_bgr) -> list[YoloBox]:
        result = self.model(frame_bgr, verbose=False, device=self.device)[0]
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


# ---------- Depth Anything V2 Small --------------------------------------

class DepthEstimator:
    """Depth Anything V2 Small via transformers ``pipeline``.

    Returns a normalized float32 HxW depth map in [0, 1] where larger values
    are closer to the camera. The raw pipeline returns a PIL ``Image`` plus a
    ``predicted_depth`` torch tensor; we use the tensor for numerical
    fidelity and normalize to the frame-level min/max.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import pipeline

        device_idx = 0 if device == "cuda" else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_idx,
        )

    def __call__(self, frame_bgr):
        import numpy as np
        from PIL import Image

        # BGR uint8 -> RGB PIL
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        out: dict[str, Any] = self.pipe(pil)

        tensor = out.get("predicted_depth")
        if tensor is None:
            depth_pil: Image.Image = out["depth"]
            depth = np.asarray(depth_pil, dtype=np.float32)
        else:
            depth = tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        # Resize to frame size if the model returned a different resolution.
        h, w = frame_bgr.shape[:2]
        if depth.shape != (h, w):
            import cv2

            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize frame-wise so the colormap has range every frame.
        lo, hi = float(depth.min()), float(depth.max())
        if hi - lo > 1e-6:
            depth = (depth - lo) / (hi - lo)
        else:
            depth = np.zeros_like(depth)
        return depth


# ---------- SegFormer semantic segmentation ------------------------------

@dataclass
class SegRegion:
    label: str
    score: float
    mask: Any  # numpy bool HxW


class SemanticSegmenter:
    """SegFormer-B0 finetuned on ADE20K, via transformers ``pipeline``.

    Returns a list of SegRegion — one per detected semantic region — with a
    boolean mask the size of the input frame.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import pipeline

        device_idx = 0 if device == "cuda" else -1
        self.pipe = pipeline(
            task="image-segmentation",
            model="nvidia/segformer-b0-finetuned-ade-512-512",
            device=device_idx,
        )

    def __call__(self, frame_bgr) -> list[SegRegion]:
        import numpy as np
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        raw = self.pipe(pil)

        h, w = frame_bgr.shape[:2]
        regions: list[SegRegion] = []
        for item in raw:
            mask_pil: Image.Image = item["mask"]
            if mask_pil.size != (w, h):
                mask_pil = mask_pil.resize((w, h), resample=Image.NEAREST)
            mask = np.asarray(mask_pil).astype(bool)
            regions.append(
                SegRegion(
                    label=str(item.get("label", "?")),
                    score=float(item.get("score") or 0.0),
                    mask=mask,
                )
            )
        return regions
