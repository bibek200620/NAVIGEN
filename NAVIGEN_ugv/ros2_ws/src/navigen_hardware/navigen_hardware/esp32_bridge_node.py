"""ESP32 serial bridge node.

/cmd_vel (Twist) -> velocity limits + acceleration ramps -> left/right wheel
velocities -> serial frames to the ESP32. Telemetry frames back from the ESP32
are published as /motor/telemetry, /wheel/odom, /battery and /ultrasonic/*.

mock_hardware:=true simulates the ESP32 so the full stack runs with no hardware.
A stale /cmd_vel (older than command_timeout_s) is replaced by zero velocity;
the ESP32 additionally enforces its own on-board watchdog.
"""
import math

import rclpy
from geometry_msgs.msg import Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Range

from navigen_interfaces.msg import MotorTelemetry
from navigen_hardware import serial_protocol as sp
from navigen_hardware.kinematics import (DiffDriveKinematics, KinematicsConfig,
                                         VelocityRamp)


class Esp32Bridge(Node):

    def __init__(self):
        super().__init__('esp32_bridge')
        d = self.declare_parameter
        d('serial_port', '/dev/ttyUSB0')
        d('baud_rate', 115200)
        d('mock_hardware', False)
        d('command_rate_hz', 30.0)
        d('command_timeout_s', 0.3)
        d('wheel_radius', 0.0625)
        d('track_width', 0.34)
        d('ticks_per_revolution', 0.0)
        d('max_linear_velocity', 0.4)
        d('max_angular_velocity', 1.0)
        d('max_wheel_velocity', 0.6)
        d('max_linear_accel', 0.8)
        d('max_angular_accel', 2.0)
        d('ultrasonic_fov_rad', 0.26)
        d('ultrasonic_max_range', 4.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.kin = DiffDriveKinematics(KinematicsConfig(
            wheel_radius=p('wheel_radius'), track_width=p('track_width'),
            max_linear_velocity=p('max_linear_velocity'),
            max_angular_velocity=p('max_angular_velocity'),
            max_wheel_velocity=p('max_wheel_velocity')))
        self.ramp_v = VelocityRamp(p('max_linear_accel'))
        self.ramp_w = VelocityRamp(p('max_angular_accel'))
        self.cmd_timeout = p('command_timeout_s')
        self.command_period = 1.0 / p('command_rate_hz')
        self.mock = p('mock_hardware')
        self.us_fov = p('ultrasonic_fov_rad')
        self.us_max = p('ultrasonic_max_range')
        self.ticks_per_rev = p('ticks_per_revolution')

        self.parser = sp.FrameParser()
        self.last_cmd = Twist()
        self.last_cmd_time = -1.0
        self.setpoints = (0.0, 0.0)
        self.pose = [0.0, 0.0, 0.0]  # x, y, yaw integrated from telemetry
        self.last_telemetry_time = None
        self._mock_distance = [0.0, 0.0]

        self.serial = None
        if not self.mock:
            import serial  # lazy: mock mode needs no pyserial
            self.serial = serial.Serial(p('serial_port'), p('baud_rate'), timeout=0)
            self.get_logger().info(f"connected to ESP32 on {p('serial_port')}")
        else:
            self.get_logger().warn('MOCK hardware mode: simulating the ESP32')

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.pub_telemetry = self.create_publisher(MotorTelemetry, 'motor/telemetry', 10)
        self.pub_odom = self.create_publisher(Odometry, 'wheel/odom', 10)
        self.pub_battery = self.create_publisher(BatteryState, 'battery', 10)
        self.pub_us_left = self.create_publisher(Range, 'ultrasonic/front_left', 10)
        self.pub_us_right = self.create_publisher(Range, 'ultrasonic/front_right', 10)

        self.create_timer(self.command_period, self.send_command)
        self.create_timer(0.02, self.poll_telemetry)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- command path ----------------
    def on_cmd_vel(self, msg: Twist):
        if not all(map(math.isfinite, (msg.linear.x, msg.angular.z))):
            self.get_logger().error('rejecting non-finite /cmd_vel')
            return
        self.last_cmd = msg
        self.last_cmd_time = self.now_s()

    def send_command(self):
        stale = self.last_cmd_time < 0 or (self.now_s() - self.last_cmd_time) > self.cmd_timeout
        target_v = 0.0 if stale else self.last_cmd.linear.x
        target_w = 0.0 if stale else self.last_cmd.angular.z
        v = self.ramp_v.step(target_v, self.command_period)
        w = self.ramp_w.step(target_w, self.command_period)
        self.setpoints = self.kin.twist_to_wheels(v, w)
        if self.serial is not None:
            self.serial.write(sp.encode_velocity_command(*self.setpoints))

    # ---------------- telemetry path ----------------
    def poll_telemetry(self):
        if self.mock:
            self.handle_telemetry(self._mock_telemetry())
            return
        data = self.serial.read(256)
        if not data:
            return
        for msg_id, payload in self.parser.feed(data):
            if msg_id == sp.MSG_TELEMETRY:
                try:
                    self.handle_telemetry(sp.decode_telemetry(payload))
                except ValueError as exc:
                    self.get_logger().warn(f'bad telemetry: {exc}')

    def _mock_telemetry(self) -> sp.Telemetry:
        left, right = self.setpoints  # perfect tracking
        for i, vel in enumerate((left, right)):
            self._mock_distance[i] += vel * 0.02
        to_ticks = (self.ticks_per_rev / (2.0 * math.pi * self.kin.config.wheel_radius)
                    if self.ticks_per_rev > 0 else 0.0)
        return sp.Telemetry(
            left_velocity=left, right_velocity=right, left_pwm=0, right_pwm=0,
            left_ticks=int(self._mock_distance[0] * to_ticks),
            right_ticks=int(self._mock_distance[1] * to_ticks),
            battery_voltage=12.0, ultrasonic_left=2.5, ultrasonic_right=2.5,
            estop_active=False, watchdog_triggered=False)

    def handle_telemetry(self, t: sp.Telemetry):
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9

        msg = MotorTelemetry()
        msg.header.stamp = now.to_msg()
        msg.left_velocity, msg.right_velocity = t.left_velocity, t.right_velocity
        msg.left_setpoint, msg.right_setpoint = self.setpoints
        msg.left_pwm, msg.right_pwm = t.left_pwm, t.right_pwm
        msg.left_ticks, msg.right_ticks = t.left_ticks, t.right_ticks
        msg.battery_voltage = t.battery_voltage
        msg.ultrasonic_left, msg.ultrasonic_right = t.ultrasonic_left, t.ultrasonic_right
        msg.estop_active = t.estop_active
        msg.watchdog_triggered = t.watchdog_triggered
        self.pub_telemetry.publish(msg)

        # integrate wheel odometry
        if self.last_telemetry_time is not None:
            dt = now_s - self.last_telemetry_time
            if 0.0 < dt < 1.0:
                v, w = self.kin.wheels_to_twist(t.left_velocity, t.right_velocity)
                self.pose[2] += w * dt
                self.pose[0] += v * math.cos(self.pose[2]) * dt
                self.pose[1] += v * math.sin(self.pose[2]) * dt
                self.publish_odom(now, v, w)
        self.last_telemetry_time = now_s

        batt = BatteryState()
        batt.header.stamp = now.to_msg()
        batt.voltage = t.battery_voltage
        batt.present = True
        self.pub_battery.publish(batt)

        for pub, frame, dist in ((self.pub_us_left, 'ultrasonic_left_link', t.ultrasonic_left),
                                 (self.pub_us_right, 'ultrasonic_right_link', t.ultrasonic_right)):
            r = Range()
            r.header.stamp = now.to_msg()
            r.header.frame_id = frame
            r.radiation_type = Range.ULTRASOUND
            r.field_of_view = self.us_fov
            r.min_range = 0.02
            r.max_range = self.us_max
            r.range = dist if dist >= 0.0 else float('inf')
            pub.publish(r)

    def publish_odom(self, now, v: float, w: float):
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.pose[0]
        odom.pose.pose.position.y = self.pose[1]
        odom.pose.pose.orientation = Quaternion(
            z=math.sin(self.pose[2] / 2.0), w=math.cos(self.pose[2] / 2.0))
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        # conservative fixed covariance; refined during Phase 6 EKF tuning
        odom.pose.covariance[0] = odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.1
        odom.twist.covariance[0] = 0.02
        odom.twist.covariance[35] = 0.05
        self.pub_odom.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = Esp32Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial is not None:
            node.serial.write(sp.encode_velocity_command(0.0, 0.0))
            node.serial.close()
        node.destroy_node()


if __name__ == '__main__':
    main()
