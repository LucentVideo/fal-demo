"""Three perceptual model wrappers, one class each. No fal imports.

Each wrapper takes a numpy BGR uint8 frame and returns a small, uniform
python-native output that ``compose.py`` knows how to render. Heavy imports
(torch, transformers, ultralytics) live inside __init__ so this module is
cheap to import on the fal runner's spec pass and in local smoke tests.

The Depth and Seg wrappers call the underlying model directly rather than
going through ``transformers.pipeline()`` — the pipeline adds 3-7x overhead
from CPU-side postprocessing that blows the realtime frame budget.
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


# ---------- Depth Anything V2 Small --------------------------------------

class DepthEstimator:
    """Depth Anything V2 Small — direct model call, no pipeline overhead.

    Returns a normalized float32 HxW depth map in [0, 1] where larger values
    are closer to the camera.
    """

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        from core import detect_device

        self.device = device or detect_device()
        self._torch_device = torch.device(self.device)
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self._use_fp16 = self.device == "cuda"
        dtype = torch.float16 if self._use_fp16 else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, dtype=dtype,
        ).to(self._torch_device).eval()
        if self.device == "cuda":
            self.model = torch.compile(self.model, mode="max-autotune")

    def __call__(self, frame_bgr):
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device=self._torch_device, dtype=torch.float16 if self._use_fp16 else torch.float32)
                  if v.is_floating_point() else v.to(self._torch_device)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            depth_tensor = self.model(**inputs).predicted_depth

        depth = depth_tensor.squeeze().float().cpu().numpy()

        h, w = frame_bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

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
    """SegFormer-B0 finetuned on ADE20K — direct model call, no pipeline overhead.

    Returns a list of SegRegion — one per detected semantic class — with a
    boolean mask the size of the input frame. The pipeline version does
    expensive per-instance mask extraction in Python; this version operates
    on the raw logits argmax which is ~7x faster.
    """

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        from core import detect_device

        self.device = device or detect_device()
        self._torch_device = torch.device(self.device)
        self._use_fp16 = self.device == "cuda"
        dtype = torch.float16 if self._use_fp16 else torch.float32
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_id, dtype=dtype,
        ).to(self._torch_device).eval()
        if self.device == "cuda":
            self.model = torch.compile(self.model, mode="max-autotune")
        self._id2label = self.model.config.id2label

    def __call__(self, frame_bgr) -> list[SegRegion]:
        import numpy as np
        import torch
        from PIL import Image

        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device=self._torch_device, dtype=torch.float16 if self._use_fp16 else torch.float32)
                  if v.is_floating_point() else v.to(self._torch_device)
                  for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        h, w = frame_bgr.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

        present_ids = np.unique(seg_map)
        regions: list[SegRegion] = []
        for cls_id in present_ids:
            mask = seg_map == cls_id
            label = self._id2label.get(int(cls_id), str(cls_id))
            pixel_fraction = float(mask.sum()) / (h * w)
            regions.append(SegRegion(label=label, score=pixel_fraction, mask=mask))
        return regions
