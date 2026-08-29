"""Deterministic mock backend for tests and early simulation phases.

Marks a widening ground trapezoid in front of the robot TRAVERSABLE, the top
third NON_TRAVERSABLE (sky / far field) and everything else UNKNOWN.
"""
import numpy as np

from navigen_perception.backends import base


class MockSegmentationBackend(base.SegmentationBackend):

    name = 'mock'

    def infer(self, image_bgr: np.ndarray) -> base.SegmentationResult:
        h, w = image_bgr.shape[:2]
        mask = np.full((h, w), base.UNKNOWN, dtype=np.uint8)
        horizon = h // 3
        mask[:horizon, :] = base.NON_TRAVERSABLE
        for row in range(horizon, h):
            frac = (row - horizon) / max(1, h - horizon)
            half = int((0.15 + 0.35 * frac) * w)
            mask[row, max(0, w // 2 - half):min(w, w // 2 + half)] = base.TRAVERSABLE
        return base.SegmentationResult(mask=mask, confidence=1.0)
