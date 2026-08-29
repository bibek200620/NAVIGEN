import pytest

from navigen_hardware import serial_protocol as sp


def make_telemetry(**overrides):
    values = dict(left_velocity=0.25, right_velocity=-0.25, left_pwm=120,
                  right_pwm=-120, left_ticks=12345, right_ticks=-6789,
                  battery_voltage=11.7, ultrasonic_left=1.234,
                  ultrasonic_right=-1.0, estop_active=False,
                  watchdog_triggered=True)
    values.update(overrides)
    return sp.Telemetry(**values)


def test_velocity_command_round_trip_frame():
    frame = sp.encode_velocity_command(0.25, -0.25)
    parser = sp.FrameParser()
    frames = parser.feed(frame)
    assert frames == [(sp.MSG_CMD_VELOCITY, b'\xfa\x00\x06\xff')]


def test_telemetry_round_trip():
    t = make_telemetry()
    frame = sp.encode_telemetry(t)
    parser = sp.FrameParser()
    ((msg_id, payload),) = parser.feed(frame)
    assert msg_id == sp.MSG_TELEMETRY
    decoded = sp.decode_telemetry(payload)
    assert decoded.left_velocity == pytest.approx(0.25)
    assert decoded.right_velocity == pytest.approx(-0.25)
    assert decoded.left_ticks == 12345
    assert decoded.right_ticks == -6789
    assert decoded.battery_voltage == pytest.approx(11.7)
    assert decoded.ultrasonic_left == pytest.approx(1.234)
    assert decoded.ultrasonic_right == -1.0  # invalid marker survives
    assert decoded.watchdog_triggered is True
    assert decoded.estop_active is False


def test_partial_feed_reassembly():
    frame = sp.encode_telemetry(make_telemetry())
    parser = sp.FrameParser()
    assert parser.feed(frame[:7]) == []
    frames = parser.feed(frame[7:])
    assert len(frames) == 1


def test_corrupted_frame_rejected_then_resync():
    good = sp.encode_telemetry(make_telemetry())
    corrupted = bytearray(good)
    corrupted[10] ^= 0xFF  # flip payload bits -> CRC mismatch
    parser = sp.FrameParser()
    assert parser.feed(bytes(corrupted)) == []
    assert parser.crc_errors == 1
    assert len(parser.feed(good)) == 1  # parser recovers


def test_noise_between_frames_ignored():
    frame = sp.encode_velocity_command(0.1, 0.1)
    parser = sp.FrameParser()
    frames = parser.feed(b'\x00garbage\xaa' + frame + b'\xff\xfe' + frame)
    assert len(frames) == 2


def test_estop_frame():
    parser = sp.FrameParser()
    ((msg_id, payload),) = parser.feed(sp.encode_estop(True))
    assert msg_id == sp.MSG_ESTOP
    assert payload == b'\x01'


def test_wrong_payload_size_raises():
    with pytest.raises(ValueError):
        sp.decode_telemetry(b'\x00\x01\x02')
