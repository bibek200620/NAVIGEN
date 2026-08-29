"""ONNX Runtime segmentation backend (populated with a trained model in Phase 7).

Expects a semantic segmentation model with:
  input : NCHW float32, normalized with the configured mean/std
  output: N x num_classes x H x W logits
Model classes are mapped to NAVIGEN class ids through class_map, so any
public or custom model can be adapted without code changes elsewhere.
"""
from __future__ import annotations

import numpy as np

from navigen_perception.backends import base


class OnnxSegmentationBackend(base.SegmentationBackend):

    name = 'onnx'

    def __init__(self, model_path: str, input_width: int = 320, input_height: int = 240,
                 class_map: list | None = None,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'onnxruntime is not installed. Install it with: pip install onnxruntime') from exc
        import cv2  # local import keeps module importable in minimal test envs
        self._cv2 = cv2
        if not model_path:
            raise RuntimeError('perception.yaml: model_path must be set for the onnx backend')
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (input_width, input_height)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        # identity map by default: model class i -> NAVIGEN id min(i, OBSTACLE)
        self.class_map = class_map

    def infer(self, image_bgr: np.ndarray) -> base.SegmentationResult:
        cv2 = self._cv2
        h, w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = (rgb.transpose(2, 0, 1) - self.mean) / self.std
        logits = self.session.run(None, {self.input_name: tensor[np.newaxis]})[0][0]

        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp / exp.sum(axis=0, keepdims=True)
        class_idx = probs.argmax(axis=0).astype(np.uint8)
        confidence = float(probs.max(axis=0).mean())

        if self.class_map:
            lut = np.asarray(self.class_map, dtype=np.uint8)
            class_idx = lut[np.clip(class_idx, 0, len(lut) - 1)]
        else:
            class_idx = np.minimum(class_idx, base.OBSTACLE)

        mask = cv2.resize(class_idx, (w, h), interpolation=cv2.INTER_NEAREST)
        return base.SegmentationResult(mask=mask, confidence=confidence)
