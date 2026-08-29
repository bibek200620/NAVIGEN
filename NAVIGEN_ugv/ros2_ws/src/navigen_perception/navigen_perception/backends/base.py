"""Segmentation backend abstraction.

Navigation code depends ONLY on this contract, so segmentation models and
inference runtimes are swappable without touching the rest of the system.

Class ids (published in /traversability/mask as mono8):
  0 UNKNOWN, 1 TRAVERSABLE, 2 NON_TRAVERSABLE, 3 OBSTACLE
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

UNKNOWN = 0
TRAVERSABLE = 1
NON_TRAVERSABLE = 2
OBSTACLE = 3

CLASS_COLORS_BGR = {
    UNKNOWN: (128, 128, 128),
    TRAVERSABLE: (0, 200, 0),
    NON_TRAVERSABLE: (0, 140, 255),
    OBSTACLE: (0, 0, 255),
}


@dataclass
class SegmentationResult:
    mask: np.ndarray      # HxW uint8 class ids (same resolution as the input image)
    confidence: float     # mean confidence in [0, 1]


class SegmentationBackend(abc.ABC):
    """Contract every traversability backend must fulfil."""

    name: str = 'abstract'

    @abc.abstractmethod
    def infer(self, image_bgr: np.ndarray) -> SegmentationResult:
        """image_bgr: HxWx3 uint8. Returns a mask at the SAME resolution."""


def colorize(mask: np.ndarray) -> np.ndarray:
    """Class-id mask -> BGR visualization image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS_BGR.items():
        out[mask == class_id] = color
    return out
