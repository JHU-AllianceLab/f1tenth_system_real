"""ROS 2 controller for a physical ego vehicle and a virtual opponent."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from .core import (
    Centerline,
    ComposedSupervisor,
    MPPIController,
    ObservationBuilder,
    PairProgressTracker,
    Pose2D,
    QCBFLineSearch,
    StartPoseOdomAligner,
    VehicleState,
    compensate_scan_for_physical_footprint,
    footprint_lidar_margin,
    inject_virtual_opponent,
    rectangle_separation,
    resample_laser_scan,
)
from .model_runtime import TorchScriptSafetyModel


def _yaw(orientation) -> float:
    siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.atan2(siny, cosy)


class ComposedControllerNode(Node):
    def __init__(self):
        super().__init__("composed_safety_controller")
        self._declare_parameters()
        bundle = Path(self.get_parameter("model_bundle").value).expanduser().resolve()
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"model_bundle does not contain manifest.json: {bundle}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_path = bundle / manifest["training_config"]
        waypoint_path = bundle / manifest["waypoints"]
        self.contract = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.centerline = Centerline.from_csv(waypoint_path)
        self.builder = ObservationBuilder(self.contract)
        device = str(self.get_parameter("torch_device").value)
        self.p1_model = TorchScriptSafetyModel(bundle, "p1", device)
        self.p2_model = TorchScriptSafetyModel(bundle, "p2", device)
        if self.p1_model.observation_size != self.builder.p1_size:
            raise ValueError("P1 model and training configuration do not match")
        if self.p2_model.observation_size != self.builder.p2_size:
            raise ValueError("P2 model and training configuration do not match")

        environment = self.contract["environment"]
        observation = self.contract["observation"]
        safety = self.contract["safety"]
        interaction = self.contract["interaction"]
        composed = self.contract["composed_safety"]
        filter_config = self.contract["filter"]
        self.lidar_fov = float(observation["lidar_fov"])
        self.max_scan_range = float(observation["max_scan_range"])
        self.trained_vehicle_length = float(safety["vehicle_length"])
        self.trained_vehicle_width = float(safety["vehicle_width"])
        self.vehicle_length = float(
            self.get_parameter("physical_vehicle_length_m").value
        )
        self.vehicle_width = float(
            self.get_parameter("physical_vehicle_width_m").value
        )
        if min(self.vehicle_length, self.vehicle_width) <= 0.0:
            raise ValueError("Physical vehicle dimensions must be positive")
        self.opponent_length = float(safety["opponent_vehicle_length"])
        self.opponent_width = float(safety["opponent_vehicle_width"])
        self.clearance = float(interaction["vehicle_clearance_m"])
        self.safety_buffer = float(safety["buffer"])
        self.max_age = float(self.get_parameter("max_input_age_sec").value)
        self.max_latency = float(self.get_parameter("max_control_latency_sec").value)
        self.shadow_mode = bool(self.get_parameter("shadow_mode").value)
        self.inject_opponent = bool(
            self.get_parameter("inject_virtual_opponent_into_scan").value
        )
        self.repeat_overtaking = bool(
            self.get_parameter("repeated_overtaking").value
        )
        self.encounter_entry_distance = float(
            self.get_parameter("encounter_entry_distance_m").value
        )
        self.minimum_forward_gap = float(
            self.get_parameter("minimum_forward_gap_m").value
        )
        if self.encounter_entry_distance <= 0.0 or self.minimum_forward_gap < 0.0:
            raise ValueError("Repeated-overtaking distance parameters are invalid")

        training_low = np.asarray(manifest["action_low"], dtype=np.float32)
        training_high = np.asarray(manifest["action_high"], dtype=np.float32)
        maximum_speed = float(self.get_parameter("max_command_speed_mps").value)
        if maximum_speed <= 0.0:
            raise ValueError("max_command_speed_mps must be positive")
        deployment_high = training_high.copy()
        deployment_high[1] = min(deployment_high[1], maximum_speed)
        filter_kwargs = dict(
            action_low=training_low,
            action_high=deployment_high,
            gamma=float(filter_config["gamma"]),
            threshold=float(filter_config["threshold"]),
            points=int(filter_config["line_search_points"]),
            minimum_alpha=float(filter_config["minimum_alpha"]),
            disagreement_threshold=float(
                filter_config["critic_disagreement_threshold"]
            ),
        )
        p1_filter = QCBFLineSearch(self.p1_model, **filter_kwargs)
        p2_filter = QCBFLineSearch(self.p2_model, **filter_kwargs)
        self.supervisor = ComposedSupervisor(
            p1_filter,
            p2_filter,
            epsilon_enter=float(composed["epsilon_enter"]),
            epsilon_emergency=float(composed["epsilon_emergency"]),
            confirm_steps=int(composed["handoff_confirm_steps"]),
        )
        deployment_mppi = dict(self.contract["mppi"])
        deployment_mppi["wheelbase"] = float(
            self.get_parameter("mppi_wheelbase_m").value
        )
        deployment_mppi["speed_max"] = float(deployment_high[1])
        self.mppi = MPPIController(
            self.centerline,
            deployment_mppi,
            environment,
            seed=int(self.get_parameter("mppi_seed").value),
        )
        self.progress = PairProgressTracker(self.centerline)
        start = self.get_parameter("map_start_pose").value
        if len(start) != 3:
            raise ValueError("map_start_pose must contain [x, y, yaw]")
        self.aligner = StartPoseOdomAligner(Pose2D(*map(float, start)))
        self.ego_pose_is_map_frame = bool(
            self.get_parameter("ego_pose_is_map_frame").value
        )
        self.previous_action = np.asarray([0.0, 0.0], dtype=np.float32)
        self.enabled = bool(self.get_parameter("start_enabled").value)
        self.scan_message = None
        self.scan_received = None
        self.ego_state = None
        self.ego_received = None
        self.opponent_state = None
        self.opponent_received = None
        self.encounter_count = 1
        self.completed_encounters = 0

        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            str(self.get_parameter("drive_topic").value),
            10,
        )
        self.shadow_publisher = self.create_publisher(
            AckermannDriveStamped,
            str(self.get_parameter("shadow_drive_topic").value),
            10,
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("ego_odom_topic").value),
            self._ego_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("opponent_odom_topic").value),
            self._opponent_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("enable_topic").value),
            self._enable_callback,
            10,
        )
        rate = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / rate, self._control_step)
        self.get_logger().warn(
            "Controller initialized {} and {} at speed limit {:.2f} m/s; enabled={}".format(
                "in SHADOW mode" if self.shadow_mode else "for LIVE /drive output",
                "with virtual LiDAR injection" if self.inject_opponent else "without virtual LiDAR injection",
                deployment_high[1],
                self.enabled,
            )
        )

    def _declare_parameters(self) -> None:
        defaults = {
            "model_bundle": "",
            "torch_device": "cpu",
            "control_rate_hz": 33.333333,
            "max_input_age_sec": 0.15,
            "max_control_latency_sec": 0.05,
            "max_command_speed_mps": 0.5,
            # Measured physical ego chassis; the exported model contract
            # remains 0.29 x 0.155 m and is compensated in the LiDAR input.
            "physical_vehicle_length_m": 0.568,
            "physical_vehicle_width_m": 0.296,
            "shadow_mode": True,
            "start_enabled": False,
            "inject_virtual_opponent_into_scan": True,
            "ego_pose_is_map_frame": False,
            "map_start_pose": [1.9251087, 1.3102571, -1.5716521],
            "laser_x": 0.27,
            "laser_y": 0.0,
            "laser_yaw": 0.0,
            "mppi_seed": 7,
            "mppi_wheelbase_m": 0.324,
            "repeated_overtaking": True,
            "encounter_entry_distance_m": 1.0,
            "minimum_forward_gap_m": 0.05,
            "scan_topic": "/scan",
            "ego_odom_topic": "/odom",
            "opponent_odom_topic": "/virtual_opponent/odom",
            "enable_topic": "/autonomy/enable",
            "drive_topic": "/drive",
            "shadow_drive_topic": "/safety_controller/shadow_drive",
            "status_topic": "/safety_controller/status",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _scan_callback(self, message: LaserScan) -> None:
        self.scan_message = message
        self.scan_received = self._now()

    def _odom_state(self, message: Odometry, align: bool) -> VehicleState:
        pose = Pose2D(
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            _yaw(message.pose.pose.orientation),
        )
        if align:
            pose = self.aligner.transform(pose)
        return VehicleState(
            pose,
            float(message.twist.twist.linear.x),
            float(message.twist.twist.linear.y),
            float(message.twist.twist.angular.z),
        )

    def _ego_callback(self, message: Odometry) -> None:
        self.ego_state = self._odom_state(
            message, align=not self.ego_pose_is_map_frame
        )
        self.ego_received = self._now()

    def _opponent_callback(self, message: Odometry) -> None:
        self.opponent_state = self._odom_state(message, align=False)
        self.opponent_received = self._now()

    def _enable_callback(self, message: Bool) -> None:
        requested = bool(message.data)
        if requested and not self.enabled:
            self.progress.reset()
            self.supervisor.reset()
            self.mppi.reset()
            self.previous_action[:] = 0.0
            self.encounter_count = 1
            self.completed_encounters = 0
        self.enabled = requested
        if not self.enabled:
            self._publish_stop("disabled")
        self.get_logger().warn(f"Autonomous output enabled={self.enabled}")

    def _publish(self, action: np.ndarray, publisher) -> None:
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.drive.steering_angle = float(action[0])
        message.drive.speed = float(action[1])
        publisher.publish(message)

    def _publish_status(self, **values) -> None:
        message = String()
        message.data = json.dumps(values, sort_keys=True)
        self.status_publisher.publish(message)

    def _publish_stop(self, reason: str) -> None:
        stop = np.zeros(2, dtype=np.float32)
        self.previous_action = stop
        if not self.shadow_mode:
            self._publish(stop, self.drive_publisher)
        self._publish_status(enabled=self.enabled, ready=False, reason=reason)

    def _control_step(self) -> None:
        now = self._now()
        inputs = (
            ("scan", self.scan_received),
            ("ego", self.ego_received),
            ("opponent", self.opponent_received),
        )
        missing = [name for name, stamp in inputs if stamp is None]
        stale = [name for name, stamp in inputs if stamp is not None and now - stamp > self.max_age]
        if missing or stale:
            self._publish_stop(
                "missing:" + ",".join(missing)
                if missing
                else "stale:" + ",".join(stale)
            )
            return
        if not self.enabled:
            self._publish_stop("disabled")
            return

        physical_scan, ray_angles = resample_laser_scan(
            self.scan_message.ranges,
            float(self.scan_message.angle_min),
            float(self.scan_message.angle_increment),
            beams=self.builder.lidar_beams,
            field_of_view=self.lidar_fov,
            max_range=self.max_scan_range,
            laser_x=float(self.get_parameter("laser_x").value),
            laser_y=float(self.get_parameter("laser_y").value),
            laser_yaw=float(self.get_parameter("laser_yaw").value),
        )
        wall_scan = physical_scan.copy()
        if self.inject_opponent:
            physical_scan = inject_virtual_opponent(
                physical_scan,
                ray_angles,
                self.ego_state.pose,
                self.opponent_state.pose,
                self.opponent_length,
                self.opponent_width,
            )
        scan = compensate_scan_for_physical_footprint(
            physical_scan,
            ray_angles,
            trained_length=self.trained_vehicle_length,
            trained_width=self.trained_vehicle_width,
            physical_length=self.vehicle_length,
            physical_width=self.vehicle_width,
        )
        ego_progress, opponent_progress = self.progress.update(
            self.ego_state.pose, self.opponent_state.pose
        )
        if self.repeat_overtaking and self.supervisor.phase == "p1_stay_safe":
            ego_wrapped = self.centerline.wrapped_progress(self.ego_state.pose)
            opponent_wrapped = self.centerline.wrapped_progress(
                self.opponent_state.pose
            )
            forward_gap = (
                opponent_wrapped - ego_wrapped
            ) % self.centerline.length
            if (
                self.minimum_forward_gap
                <= forward_gap
                <= self.encounter_entry_distance
            ):
                ego_progress, opponent_progress = (
                    self.progress.retarget_opponent_ahead(forward_gap)
                )
                self.supervisor.rearm_p2()
                self.encounter_count += 1
        p1_observation, _ = self.builder.p1(
            scan,
            self.ego_state,
            self.opponent_state,
            self.previous_action,
            ego_progress,
            opponent_progress,
        )
        p1_value = self.p1_model.value(p1_observation)
        p2_observation, task_lead = self.builder.p2(
            p1_observation, p1_value, ego_progress, opponent_progress
        )
        wall_margin = footprint_lidar_margin(
            wall_scan,
            ray_angles,
            self.vehicle_length,
            self.vehicle_width,
            self.safety_buffer,
        )
        opponent_margin = rectangle_separation(
            self.ego_state.pose,
            self.opponent_state.pose,
            self.vehicle_length,
            self.vehicle_width,
            self.opponent_length,
            self.opponent_width,
            self.clearance,
        )
        g_safe = min(wall_margin, opponent_margin)
        nominal = self.mppi.plan(self.ego_state)
        result = self.supervisor.step(
            p1_observation,
            p2_observation,
            nominal,
            p1_value,
            task_lead,
            g_safe,
        )
        if result.handoff:
            self.completed_encounters += 1
        filtered = result.filter_result
        if not np.isfinite(result.action).all():
            self._publish_stop("non_finite_controller_output")
            return
        latency = self._now() - now
        if latency > self.max_latency:
            self._publish_stop(
                "control_deadline:{:.1f}ms>{:.1f}ms".format(
                    1000.0 * latency, 1000.0 * self.max_latency
                )
            )
            return
        self.previous_action = result.action.copy()
        self.mppi.previous_action = result.action.astype(np.float64)
        self._publish(result.action, self.shadow_publisher)
        if not self.shadow_mode:
            self._publish(result.action, self.drive_publisher)
        self._publish_status(
            enabled=True,
            ready=True,
            shadow_mode=self.shadow_mode,
            phase=result.phase,
            handoff=result.handoff,
            emergency=result.emergency,
            intervened=filtered.intervened,
            feasible=filtered.feasible,
            fallback=filtered.fallback,
            steering_rad=float(result.action[0]),
            speed_mps=float(result.action[1]),
            p1_value=p1_value,
            p2_value=(
                filtered.q_safe if result.phase == "p2_reach_avoid" else None
            ),
            control_latency_ms=1000.0 * latency,
            lead_margin_m=task_lead,
            wall_margin_m=wall_margin,
            opponent_margin_m=opponent_margin,
            g_safe_m=g_safe,
            q_nominal=filtered.q_nominal,
            q_safe=filtered.q_safe,
            q_required=filtered.required_q,
            encounter_count=self.encounter_count,
            completed_encounters=self.completed_encounters,
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ComposedControllerNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node._publish_stop("shutdown")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
