import math

import pytest

from navigen_hardware.kinematics import (DiffDriveKinematics, KinematicsConfig,
                                         VelocityRamp)

CFG = KinematicsConfig(wheel_radius=0.0625, track_width=0.34,
                       max_linear_velocity=0.4, max_angular_velocity=1.0,
                       max_wheel_velocity=0.6)


def test_straight_line():
    kin = DiffDriveKinematics(CFG)
    left, right = kin.twist_to_wheels(0.3, 0.0)
    assert left == pytest.approx(0.3)
    assert right == pytest.approx(0.3)


def test_rotation_in_place():
    kin = DiffDriveKinematics(CFG)
    left, right = kin.twist_to_wheels(0.0, 1.0)
    assert left == pytest.approx(-0.17)
    assert right == pytest.approx(0.17)


def test_body_velocity_clamped():
    kin = DiffDriveKinematics(CFG)
    left, right = kin.twist_to_wheels(5.0, 0.0)
    assert left == pytest.approx(0.4)
    assert right == pytest.approx(0.4)


def test_wheel_limit_preserves_curvature():
    cfg = KinematicsConfig(wheel_radius=0.0625, track_width=0.34,
                           max_linear_velocity=1.0, max_angular_velocity=4.0,
                           max_wheel_velocity=0.5)
    kin = DiffDriveKinematics(cfg)
    left, right = kin.twist_to_wheels(0.6, 2.0)
    assert max(abs(left), abs(right)) == pytest.approx(0.5)
    v_in, w_in = 0.6, 2.0
    v_out, w_out = kin.wheels_to_twist(left, right)
    assert v_out / w_out == pytest.approx(v_in / w_in)  # same curvature


def test_non_finite_rejected():
    kin = DiffDriveKinematics(CFG)
    with pytest.raises(ValueError):
        kin.twist_to_wheels(float('nan'), 0.0)
    with pytest.raises(ValueError):
        kin.twist_to_wheels(0.0, float('inf'))


def test_round_trip():
    kin = DiffDriveKinematics(CFG)
    left, right = kin.twist_to_wheels(0.2, 0.5)
    v, w = kin.wheels_to_twist(left, right)
    assert v == pytest.approx(0.2)
    assert w == pytest.approx(0.5)


def test_ticks_to_velocity():
    kin = DiffDriveKinematics(CFG)
    # one full revolution in one second -> circumference per second
    v = kin.ticks_to_velocity(1320.0, 1.0, 1320.0)
    assert v == pytest.approx(2.0 * math.pi * 0.0625)


def test_ticks_requires_configuration():
    kin = DiffDriveKinematics(CFG)
    with pytest.raises(ValueError):
        kin.ticks_to_distance(100, 0.0)
    with pytest.raises(ValueError):
        kin.ticks_to_velocity(100, 0.0, 1320.0)


def test_velocity_ramp_limits_acceleration():
    ramp = VelocityRamp(max_accel=1.0)
    assert ramp.step(1.0, 0.1) == pytest.approx(0.1)
    assert ramp.step(1.0, 0.1) == pytest.approx(0.2)
    ramp.reset()
    assert ramp.step(-1.0, 0.5) == pytest.approx(-0.5)
