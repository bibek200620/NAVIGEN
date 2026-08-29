"""Pure-Python safety primitives (unit-testable, no ROS imports)."""
from __future__ import annotations

import math


def command_is_valid(linear_x: float, angular_z: float) -> bool:
    """Reject NaN/Inf motion commands."""
    return math.isfinite(linear_x) and math.isfinite(angular_z)


class DebouncedThreshold:
    """Trigger only after N consecutive readings below a distance threshold.

    Filters single noisy ultrasonic readings while still reacting fast
    (N=3 at 30 Hz telemetry means ~100 ms reaction). Invalid readings (< 0)
    keep the previous decision.
    """

    def __init__(self, threshold: float, consecutive: int = 3):
        self.threshold = threshold
        self.consecutive = consecutive
        self._count = 0

    def update(self, reading: float) -> bool:
        if reading >= 0.0:
            if reading < self.threshold:
                self._count = min(self._count + 1, self.consecutive)
            else:
                self._count = 0
        return self.triggered

    @property
    def triggered(self) -> bool:
        return self._count >= self.consecutive


class StaleMonitor:
    """Track whether a data source has gone stale."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self._last: float | None = None

    def update(self, stamp_s: float) -> None:
        self._last = stamp_s

    def is_stale(self, now_s: float) -> bool:
        return self._last is None or (now_s - self._last) > self.timeout_s
