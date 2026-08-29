"""Safety supervisor.

Sits between navigation and the hardware: /cmd_vel_nav -> checks -> /cmd_vel.
Safety ALWAYS overrides navigation. Emergency triggers:
  - software e-stop (/safety/e_stop) or hardware e-stop (telemetry flag)
  - ESP32 telemetry stale (require_telemetry)
  - camera frames stale (require_camera, Phase 7)
  - SLAM LOST/stale (require_slam, Phase 8)
  - debounced ultrasonic reading below the emergency distance
Non-emergency: a stale or invalid /cmd_vel_nav simply results in zero output.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Bool

from navigen_interfaces.msg import MotorTelemetry, SafetyState, TrackingState
from navigen_safety.checks import DebouncedThreshold, StaleMonitor, command_is_valid


class SafetySupervisor(Node):

    def __init__(self):
        super().__init__('safety_supervisor')
        d = self.declare_parameter
        d('rate_hz', 20.0)
        d('max_linear_velocity', 0.4)
        d('max_angular_velocity', 1.0)
        d('cmd_timeout_s', 0.5)
        d('telemetry_timeout_s', 0.6)
        d('camera_timeout_s', 1.0)
        d('slam_timeout_s', 1.0)
        d('ultrasonic_emergency_distance', 0.35)
        d('ultrasonic_consecutive_readings', 3)
        d('require_telemetry', False)  # enable in Phase 5 (real hardware)
        d('require_camera', False)     # enable in Phase 7
        d('require_slam', False)       # enable in Phase 8

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.max_v = p('max_linear_velocity')
        self.max_w = p('max_angular_velocity')
        self.require_telemetry = p('require_telemetry')
        self.require_camera = p('require_camera')
        self.require_slam = p('require_slam')

        self.cmd_monitor = StaleMonitor(p('cmd_timeout_s'))
        self.telemetry_monitor = StaleMonitor(p('telemetry_timeout_s'))
        self.camera_monitor = StaleMonitor(p('camera_timeout_s'))
        self.slam_monitor = StaleMonitor(p('slam_timeout_s'))
        self.us_left = DebouncedThreshold(p('ultrasonic_emergency_distance'),
                                          p('ultrasonic_consecutive_readings'))
        self.us_right = DebouncedThreshold(p('ultrasonic_emergency_distance'),
                                           p('ultrasonic_consecutive_readings'))

        self.software_estop = False
        self.hardware_estop = False
        self.watchdog_triggered = False
        self.slam_state = TrackingState.INITIALIZING
        self.last_cmd = Twist()

        self.create_subscription(Twist, 'cmd_vel_nav', self.on_cmd, 10)
        self.create_subscription(Bool, 'safety/e_stop', self.on_estop, 10)
        self.create_subscription(MotorTelemetry, 'motor/telemetry', self.on_telemetry, 10)
        self.create_subscription(Range, 'ultrasonic/front_left', self.on_us_left, 10)
        self.create_subscription(Range, 'ultrasonic/front_right', self.on_us_right, 10)
        self.create_subscription(Image, 'camera/image_raw', self.on_camera, 5)
        self.create_subscription(TrackingState, 'slam/tracking_state', self.on_slam, 10)

        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_state = self.create_publisher(SafetyState, 'safety/state', 10)
        self.create_timer(1.0 / p('rate_hz'), self.tick)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- inputs ----------------
    def on_cmd(self, msg: Twist):
        if not command_is_valid(msg.linear.x, msg.angular.z):
            self.get_logger().error('rejecting non-finite /cmd_vel_nav')
            return
        self.last_cmd = msg
        self.cmd_monitor.update(self.now_s())

    def on_estop(self, msg: Bool):
        if msg.data and not self.software_estop:
            self.get_logger().warn('SOFTWARE E-STOP ACTIVATED')
        self.software_estop = msg.data

    def on_telemetry(self, msg: MotorTelemetry):
        self.telemetry_monitor.update(self.now_s())
        self.hardware_estop = msg.estop_active
        self.watchdog_triggered = msg.watchdog_triggered

    def on_us_left(self, msg: Range):
        self.us_left.update(msg.range if msg.range != float('inf') else -1.0)

    def on_us_right(self, msg: Range):
        self.us_right.update(msg.range if msg.range != float('inf') else -1.0)

    def on_camera(self, _msg: Image):
        self.camera_monitor.update(self.now_s())

    def on_slam(self, msg: TrackingState):
        self.slam_monitor.update(self.now_s())
        self.slam_state = msg.state

    # ---------------- supervisor loop ----------------
    def emergency_triggers(self, now: float) -> list:
        triggers = []
        if self.software_estop:
            triggers.append('estop_software')
        if self.hardware_estop:
            triggers.append('estop_hardware')
        if self.require_telemetry and self.telemetry_monitor.is_stale(now):
            triggers.append('telemetry_stale')
        if self.require_camera and self.camera_monitor.is_stale(now):
            triggers.append('camera_stale')
        if self.require_slam and (self.slam_monitor.is_stale(now)
                                  or self.slam_state == TrackingState.LOST):
            triggers.append('slam_lost')
        if self.us_left.triggered:
            triggers.append('ultrasonic_left')
        if self.us_right.triggered:
            triggers.append('ultrasonic_right')
        return triggers

    def tick(self):
        now = self.now_s()
        triggers = self.emergency_triggers(now)
        state = SafetyState()
        state.header.stamp = self.get_clock().now().to_msg()
        out = Twist()

        if triggers:
            state.state = SafetyState.STATE_EMERGENCY_STOP
            state.active_triggers = triggers
            state.description = 'emergency stop: ' + ', '.join(triggers)
        elif self.cmd_monitor.is_stale(now):
            state.state = SafetyState.STATE_WARNING
            state.active_triggers = ['cmd_stale']
            state.description = 'no fresh navigation command; holding zero velocity'
        else:
            state.state = SafetyState.STATE_OK
            state.description = 'forwarding navigation commands'
            out.linear.x = max(-self.max_v, min(self.max_v, self.last_cmd.linear.x))
            out.angular.z = max(-self.max_w, min(self.max_w, self.last_cmd.angular.z))

        self.pub_cmd.publish(out)
        self.pub_state.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
