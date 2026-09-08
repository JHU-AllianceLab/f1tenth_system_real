"""Publish a centerline-following opponent as Odometry and an RViz marker."""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker

from .core import Centerline


class VirtualOpponentNode(Node):
    def __init__(self):
        super().__init__("virtual_opponent")
        defaults = {
            "model_bundle": "",
            "odom_topic": "/virtual_opponent/odom",
            "marker_topic": "/virtual_opponent/marker",
            "map_frame": "map",
            "child_frame": "virtual_opponent",
            "publish_rate_hz": 40.0,
            "speed_mps": 1.5,
            "initial_progress_m": 0.75,
            "vehicle_length": 0.29,
            "vehicle_width": 0.155,
            "enabled": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        bundle = Path(self.get_parameter("model_bundle").value).expanduser().resolve()
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.centerline = Centerline.from_csv(bundle / manifest["waypoints"])
        self.progress = float(self.get_parameter("initial_progress_m").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.previous_time = self.get_clock().now()
        self.odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), 10
        )
        self.marker_publisher = self.create_publisher(
            Marker, str(self.get_parameter("marker_topic").value), 10
        )
        self.create_service(Trigger, "~/reset", self._reset)
        self.create_service(SetBool, "~/set_enabled", self._set_enabled)
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate, self._step)

    def _reset(self, request, response):
        del request
        self.progress = float(self.get_parameter("initial_progress_m").value)
        self.previous_time = self.get_clock().now()
        response.success = True
        response.message = "Virtual opponent reset"
        return response

    def _set_enabled(self, request, response):
        self.enabled = bool(request.data)
        self.previous_time = self.get_clock().now()
        response.success = True
        response.message = f"Virtual opponent enabled={self.enabled}"
        return response

    def _step(self) -> None:
        now = self.get_clock().now()
        dt = max(0.0, min(0.2, (now - self.previous_time).nanoseconds * 1e-9))
        self.previous_time = now
        speed = float(self.get_parameter("speed_mps").value) if self.enabled else 0.0
        self.progress = (self.progress + speed * dt) % self.centerline.length
        pose = self.centerline.pose_at_progress(self.progress)
        half_yaw = 0.5 * pose.yaw

        odometry = Odometry()
        odometry.header.stamp = now.to_msg()
        odometry.header.frame_id = str(self.get_parameter("map_frame").value)
        odometry.child_frame_id = str(self.get_parameter("child_frame").value)
        odometry.pose.pose.position.x = pose.x
        odometry.pose.pose.position.y = pose.y
        odometry.pose.pose.orientation.z = math.sin(half_yaw)
        odometry.pose.pose.orientation.w = math.cos(half_yaw)
        odometry.twist.twist.linear.x = speed
        self.odom_publisher.publish(odometry)

        marker = Marker()
        marker.header = odometry.header
        marker.ns = "virtual_opponent"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = odometry.pose.pose
        marker.pose.position.z = 0.05
        marker.scale.x = float(self.get_parameter("vehicle_length").value)
        marker.scale.y = float(self.get_parameter("vehicle_width").value)
        marker.scale.z = 0.10
        marker.color.r = 0.95
        marker.color.g = 0.20
        marker.color.b = 0.10
        marker.color.a = 0.85
        self.marker_publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualOpponentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

