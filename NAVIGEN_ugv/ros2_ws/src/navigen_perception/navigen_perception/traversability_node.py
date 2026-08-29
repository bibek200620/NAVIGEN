"""Traversability segmentation node.

Subscribes /camera/image_raw, runs the configured SegmentationBackend at a
reduced inference resolution (Raspberry Pi friendly) and publishes:
  /traversability/mask         mono8 class ids (0-3, see backends.base)
  /traversability/debug_image  color overlay for RViz / demos
  /traversability/confidence   std_msgs/Float32
  /traversability/fps          std_msgs/Float32 (EMA of processing rate)
"""
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from navigen_perception.backends import base
from navigen_perception.backends.mock_backend import MockSegmentationBackend


class TraversabilityNode(Node):

    def __init__(self):
        super().__init__('traversability_node')
        d = self.declare_parameter
        d('backend', 'mock')            # mock | onnx
        d('model_path', '')
        d('inference_width', 320)
        d('inference_height', 240)
        d('publish_debug', True)
        d('class_map', [0, 1, 2, 3])

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.size = (int(p('inference_width')), int(p('inference_height')))
        self.publish_debug = p('publish_debug')
        self.backend = self._make_backend(p('backend'), p('model_path'), list(p('class_map')))
        self.get_logger().info(f'segmentation backend: {self.backend.name} @ {self.size}')

        self.bridge = CvBridge()
        self._fps = 0.0
        self._last_t = None

        self.create_subscription(Image, 'camera/image_raw', self.on_image, 1)
        self.pub_mask = self.create_publisher(Image, 'traversability/mask', 2)
        self.pub_debug = self.create_publisher(Image, 'traversability/debug_image', 2)
        self.pub_conf = self.create_publisher(Float32, 'traversability/confidence', 2)
        self.pub_fps = self.create_publisher(Float32, 'traversability/fps', 2)

    def _make_backend(self, name: str, model_path: str, class_map: list):
        if name == 'mock':
            return MockSegmentationBackend()
        if name == 'onnx':
            from navigen_perception.backends.onnx_backend import OnnxSegmentationBackend
            return OnnxSegmentationBackend(model_path, self.size[0], self.size[1],
                                           class_map=class_map)
        raise ValueError(f'unknown segmentation backend: {name!r}')

    def on_image(self, msg: Image):
        start = time.monotonic()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)

        result = self.backend.infer(small)

        mask_msg = self.bridge.cv2_to_imgmsg(result.mask, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)
        self.pub_conf.publish(Float32(data=float(result.confidence)))

        if self.publish_debug:
            overlay = cv2.addWeighted(small, 0.6, base.colorize(result.mask), 0.4, 0.0)
            debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)

        dt = time.monotonic() - start
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = inst if self._last_t is None else 0.9 * self._fps + 0.1 * inst
        self._last_t = start
        self.pub_fps.publish(Float32(data=float(self._fps)))


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
