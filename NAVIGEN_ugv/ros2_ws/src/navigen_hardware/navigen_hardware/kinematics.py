"""Skid-steer / differential-drive kinematics.

Pure Python (no ROS imports) so it is unit-testable and reusable.
All tunables come from configuration (hardware.yaml) - nothing hard-coded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KinematicsConfig:
    wheel_radius: float                 # [m]
    track_width: float                  # [m] effective left-right wheel distance
    max_linear_velocity: float = 0.4    # [m/s]
    max_angular_velocity: float = 1.0   # [rad/s]
    max_wheel_velocity: float = 0.6     # [m/s]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DiffDriveKinematics:
    """Convert body twists to per-side wheel velocities and back."""

    def __init__(self, config: KinematicsConfig):
        if config.wheel_radius <= 0.0 or config.track_width <= 0.0:
            raise ValueError('wheel_radius and track_width must be positive')
        self.config = config

    def twist_to_wheels(self, linear_x: float, angular_z: float) -> tuple[float, float]:
        """Return (left, right) wheel surface velocities in m/s.

        Raises ValueError on NaN/Inf input. Applies body velocity limits, then
        proportionally rescales both sides if a wheel would exceed
        max_wheel_velocity so the commanded curvature is preserved.
        """
        if not (math.isfinite(linear_x) and math.isfinite(angular_z)):
            raise ValueError('non-finite twist command')
        cfg = self.config
        v = clamp(linear_x, -cfg.max_linear_velocity, cfg.max_linear_velocity)
        w = clamp(angular_z, -cfg.max_angular_velocity, cfg.max_angular_velocity)
        left = v - w * cfg.track_width / 2.0
        right = v + w * cfg.track_width / 2.0
        peak = max(abs(left), abs(right))
        if cfg.max_wheel_velocity > 0.0 and peak > cfg.max_wheel_velocity:
            scale = cfg.max_wheel_velocity / peak
            left *= scale
            right *= scale
        return left, right

    def wheels_to_twist(self, left: float, right: float) -> tuple[float, float]:
        """Return (linear_x, angular_z) from per-side wheel velocities."""
        return (left + right) / 2.0, (right - left) / self.config.track_width

    def ticks_to_distance(self, delta_ticks: float, ticks_per_revolution: float) -> float:
        if ticks_per_revolution <= 0.0:
            raise ValueError('ticks_per_revolution must be positive (set it in hardware.yaml)')
        return delta_ticks / ticks_per_revolution * 2.0 * math.pi * self.config.wheel_radius

    def ticks_to_velocity(self, delta_ticks: float, dt: float,
                          ticks_per_revolution: float) -> float:
        if dt <= 0.0:
            raise ValueError('dt must be positive')
        return self.ticks_to_distance(delta_ticks, ticks_per_revolution) / dt


class VelocityRamp:
    """First-order acceleration limiter for one velocity channel."""

    def __init__(self, max_accel: float):
        self.max_accel = max_accel
        self.value = 0.0

    def reset(self, value: float = 0.0) -> None:
        self.value = value

    def step(self, target: float, dt: float) -> float:
        if dt <= 0.0:
            return self.value
        max_delta = self.max_accel * dt
        self.value += clamp(target - self.value, -max_delta, max_delta)
        return self.value
