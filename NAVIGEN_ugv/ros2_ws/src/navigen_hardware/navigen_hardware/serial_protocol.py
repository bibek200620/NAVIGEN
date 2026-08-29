"""Framed binary serial protocol between the Raspberry Pi and the ESP32.

Frame: 0xAA 0x55 | msg_id (u8) | payload_len (u8) | payload | crc8
crc8 (poly 0x07, init 0x00) covers msg_id, payload_len and payload.

Pi -> ESP32:
  MSG_CMD_VELOCITY (0x01): int16 left mm/s, int16 right mm/s
  MSG_ESTOP        (0x02): uint8 active
ESP32 -> Pi:
  MSG_TELEMETRY    (0x10): int16 left mm/s, int16 right mm/s, int16 left pwm,
                           int16 right pwm, int32 left ticks, int32 right ticks,
                           uint16 battery mV, uint16 us_left mm, uint16 us_right mm,
                           uint8 flags (bit0 estop, bit1 watchdog)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

SYNC = b'\xaa\x55'
MSG_CMD_VELOCITY = 0x01
MSG_ESTOP = 0x02
MSG_TELEMETRY = 0x10

FLAG_ESTOP = 0x01
FLAG_WATCHDOG = 0x02
US_INVALID = 0xFFFF

_VEL_STRUCT = struct.Struct('<hh')
_TELEMETRY_STRUCT = struct.Struct('<hhhhiiHHHB')


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_frame(msg_id: int, payload: bytes = b'') -> bytes:
    body = bytes((msg_id, len(payload))) + payload
    return SYNC + body + bytes((crc8(body),))


def _mmps(mps: float) -> int:
    return max(-32768, min(32767, int(round(mps * 1000.0))))


def encode_velocity_command(left_mps: float, right_mps: float) -> bytes:
    return encode_frame(MSG_CMD_VELOCITY, _VEL_STRUCT.pack(_mmps(left_mps), _mmps(right_mps)))


def encode_estop(active: bool) -> bytes:
    return encode_frame(MSG_ESTOP, bytes((1 if active else 0,)))


@dataclass
class Telemetry:
    left_velocity: float        # [m/s]
    right_velocity: float       # [m/s]
    left_pwm: int
    right_pwm: int
    left_ticks: int
    right_ticks: int
    battery_voltage: float      # [V]
    ultrasonic_left: float      # [m], -1.0 invalid
    ultrasonic_right: float     # [m], -1.0 invalid
    estop_active: bool
    watchdog_triggered: bool


def encode_telemetry(t: Telemetry) -> bytes:
    """Used by unit tests and the mock; the real encoder runs on the ESP32."""
    flags = (FLAG_ESTOP if t.estop_active else 0) | (FLAG_WATCHDOG if t.watchdog_triggered else 0)
    us_l = US_INVALID if t.ultrasonic_left < 0 else min(US_INVALID - 1, int(t.ultrasonic_left * 1000))
    us_r = US_INVALID if t.ultrasonic_right < 0 else min(US_INVALID - 1, int(t.ultrasonic_right * 1000))
    payload = _TELEMETRY_STRUCT.pack(
        _mmps(t.left_velocity), _mmps(t.right_velocity), t.left_pwm, t.right_pwm,
        t.left_ticks, t.right_ticks, int(t.battery_voltage * 1000), us_l, us_r, flags)
    return encode_frame(MSG_TELEMETRY, payload)


def decode_telemetry(payload: bytes) -> Telemetry:
    if len(payload) != _TELEMETRY_STRUCT.size:
        raise ValueError('telemetry payload has wrong size')
    lv, rv, lp, rp, lt, rt, batt, us_l, us_r, flags = _TELEMETRY_STRUCT.unpack(payload)
    return Telemetry(
        left_velocity=lv / 1000.0, right_velocity=rv / 1000.0,
        left_pwm=lp, right_pwm=rp, left_ticks=lt, right_ticks=rt,
        battery_voltage=batt / 1000.0,
        ultrasonic_left=-1.0 if us_l == US_INVALID else us_l / 1000.0,
        ultrasonic_right=-1.0 if us_r == US_INVALID else us_r / 1000.0,
        estop_active=bool(flags & FLAG_ESTOP),
        watchdog_triggered=bool(flags & FLAG_WATCHDOG))


class FrameParser:
    """Incremental parser; tolerates noise, partial reads and corrupted frames."""

    def __init__(self):
        self._buf = bytearray()
        self.crc_errors = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        """Feed raw bytes; return a list of (msg_id, payload) for valid frames."""
        self._buf.extend(data)
        frames: list[tuple[int, bytes]] = []
        while True:
            start = self._buf.find(SYNC)
            if start < 0:
                # drop garbage, keep a possible trailing first sync byte
                del self._buf[:max(0, len(self._buf) - 1)]
                break
            if start > 0:
                del self._buf[:start]
            if len(self._buf) < 5:
                break
            payload_len = self._buf[3]
            total = 4 + payload_len + 1
            if len(self._buf) < total:
                break
            body = bytes(self._buf[2:4 + payload_len])
            if crc8(body) == self._buf[4 + payload_len]:
                frames.append((self._buf[2], bytes(self._buf[4:4 + payload_len])))
                del self._buf[:total]
            else:
                self.crc_errors += 1
                del self._buf[:2]  # resync past the bad sync word
        return frames
