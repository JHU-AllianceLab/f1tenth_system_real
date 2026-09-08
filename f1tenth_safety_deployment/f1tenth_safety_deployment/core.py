"""ROS-independent observation and control logic for physical deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

import numpy as np


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class VehicleState:
    pose: Pose2D
    longitudinal_speed: float
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0


class Centerline:
    """Nearest-waypoint progress contract used by the Gym environment."""

    def __init__(self, xy: np.ndarray):
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("Centerline must contain at least three xy points")
        if not np.isfinite(points).all():
            raise ValueError("Centerline contains non-finite coordinates")
        self.xy = points
        closed = np.vstack((points, points[0]))
        self.segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        if np.any(self.segment_lengths <= 0.0):
            raise ValueError("Centerline contains duplicate consecutive points")
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.segment_lengths[:-1])))
        self.length = float(np.sum(self.segment_lengths))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        delimiter: str = ";",
        x_index: int = 1,
        y_index: int = 2,
    ) -> "Centerline":
        table = np.loadtxt(path, delimiter=delimiter, comments="#", dtype=np.float64)
        table = np.atleast_2d(table)
        if table.shape[1] <= max(x_index, y_index):
            raise ValueError("Centerline does not contain configured xy columns")
        return cls(table[:, [x_index, y_index]])

    def nearest_index(self, pose: Pose2D) -> int:
        delta = self.xy - np.asarray([pose.x, pose.y], dtype=np.float64)
        return int(np.argmin(np.sum(delta * delta, axis=1)))

    def wrapped_progress(self, pose: Pose2D) -> float:
        return float(self.cumulative[self.nearest_index(pose)])

    def pose_at_progress(self, progress: float) -> Pose2D:
        value = float(progress) % self.length
        index = int(np.searchsorted(self.cumulative, value, side="right") - 1)
        index = min(max(index, 0), self.xy.shape[0] - 1)
        following = (index + 1) % self.xy.shape[0]
        segment_start = float(self.cumulative[index])
        offset = value - segment_start
        if index == self.xy.shape[0] - 1 and offset < 0.0:
            offset += self.length
        fraction = np.clip(offset / self.segment_lengths[index], 0.0, 1.0)
        point = self.xy[index] + fraction * (self.xy[following] - self.xy[index])
        tangent = self.xy[following] - self.xy[index]
        return Pose2D(float(point[0]), float(point[1]), math.atan2(tangent[1], tangent[0]))


class PairProgressTracker:
    """Unwrap ego and target progress while preserving an ahead target at reset."""

    def __init__(self, centerline: Centerline):
        self.centerline = centerline
        self.reset()

    def reset(self) -> None:
        self._previous_ego = None
        self._previous_opponent = None
        self.ego = None
        self.opponent = None

    def update(self, ego: Pose2D, opponent: Pose2D) -> tuple[float, float]:
        ego_wrapped = self.centerline.wrapped_progress(ego)
        opponent_wrapped = self.centerline.wrapped_progress(opponent)
        if self.ego is None:
            forward_gap = (opponent_wrapped - ego_wrapped) % self.centerline.length
            self.ego = ego_wrapped
            self.opponent = ego_wrapped + forward_gap
        else:
            self.ego += self._wrapped_delta(ego_wrapped, self._previous_ego)
            self.opponent += self._wrapped_delta(
                opponent_wrapped, self._previous_opponent
            )
        self._previous_ego = ego_wrapped
        self._previous_opponent = opponent_wrapped
        return float(self.ego), float(self.opponent)

    def retarget_opponent_ahead(self, forward_gap: float) -> tuple[float, float]:
        """Re-anchor the same opponent as a new target on a later lap."""
        if self.ego is None:
            raise RuntimeError("Progress must be updated before retargeting")
        gap = float(forward_gap)
        if not 0.0 <= gap <= self.centerline.length:
            raise ValueError("Forward gap must lie within one track lap")
        self.opponent = self.ego + gap
        return float(self.ego), float(self.opponent)

    def _wrapped_delta(self, current: float, previous: float) -> float:
        delta = current - previous
        half = 0.5 * self.centerline.length
        if delta > half:
            delta -= self.centerline.length
        elif delta < -half:
            delta += self.centerline.length
        return float(delta)


class StartPoseOdomAligner:
    """Map a local odometry frame onto a known physical start pose."""

    def __init__(self, map_start: Pose2D):
        self.map_start = map_start
        self.odom_start: Pose2D | None = None

    def reset(self) -> None:
        self.odom_start = None

    def transform(self, odom_pose: Pose2D) -> Pose2D:
        if self.odom_start is None:
            self.odom_start = odom_pose
        start = self.odom_start
        dx = odom_pose.x - start.x
        dy = odom_pose.y - start.y
        c0, s0 = math.cos(start.yaw), math.sin(start.yaw)
        local_x = c0 * dx + s0 * dy
        local_y = -s0 * dx + c0 * dy
        cm, sm = math.cos(self.map_start.yaw), math.sin(self.map_start.yaw)
        return Pose2D(
            self.map_start.x + cm * local_x - sm * local_y,
            self.map_start.y + sm * local_x + cm * local_y,
            float(self.map_start.yaw + wrap_angle(odom_pose.yaw - start.yaw)),
        )


def resample_laser_scan(
    ranges,
    angle_min: float,
    angle_increment: float,
    *,
    beams: int,
    field_of_view: float,
    max_range: float,
    laser_x: float = 0.0,
    laser_y: float = 0.0,
    laser_yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a physical laser scan to model-origin uniformly spaced rays."""

    values = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if values.size < 2 or beams < 2 or angle_increment == 0.0:
        raise ValueError("Laser scan and target beam count must contain at least two rays")
    clean = np.nan_to_num(values, nan=0.0, posinf=max_range, neginf=0.0)
    clean = np.clip(clean, 0.0, max_range)
    source_angles = angle_min + angle_increment * np.arange(values.size)
    bearing = source_angles + laser_yaw
    x = laser_x + clean * np.cos(bearing)
    y = laser_y + clean * np.sin(bearing)
    projected_angles = np.arctan2(y, x)
    projected_ranges = np.hypot(x, y)
    order = np.argsort(projected_angles)
    projected_angles = projected_angles[order]
    projected_ranges = projected_ranges[order]
    unique_angles, unique_indices = np.unique(projected_angles, return_index=True)
    target_angles = np.linspace(-0.5 * field_of_view, 0.5 * field_of_view, beams)
    sampled = np.interp(
        target_angles,
        unique_angles,
        projected_ranges[unique_indices],
        left=max_range,
        right=max_range,
    )
    return np.clip(sampled, 0.0, max_range).astype(np.float32), target_angles


def ray_rectangle_distances(
    ray_angles: np.ndarray,
    ego_pose: Pose2D,
    opponent_pose: Pose2D,
    length: float,
    width: float,
) -> np.ndarray:
    """Distance from ego/model origin to an oriented virtual rectangle."""

    if length <= 0.0 or width <= 0.0:
        raise ValueError("Virtual opponent dimensions must be positive")
    angles = np.asarray(ray_angles, dtype=np.float64)
    ray_world = ego_pose.yaw + angles
    c, s = math.cos(opponent_pose.yaw), math.sin(opponent_pose.yaw)
    origin_delta = np.asarray(
        [ego_pose.x - opponent_pose.x, ego_pose.y - opponent_pose.y]
    )
    origin_local = np.asarray(
        [c * origin_delta[0] + s * origin_delta[1], -s * origin_delta[0] + c * origin_delta[1]]
    )
    dx_world = np.cos(ray_world)
    dy_world = np.sin(ray_world)
    directions = np.column_stack(
        (c * dx_world + s * dy_world, -s * dx_world + c * dy_world)
    )
    half = np.asarray([0.5 * length, 0.5 * width])
    lower = -half
    upper = half
    t_enter = np.full(angles.shape, -np.inf)
    t_exit = np.full(angles.shape, np.inf)
    valid = np.ones(angles.shape, dtype=bool)
    for axis in range(2):
        direction = directions[:, axis]
        parallel = np.abs(direction) < 1e-10
        valid &= ~(parallel & ((origin_local[axis] < lower[axis]) | (origin_local[axis] > upper[axis])))
        safe_direction = np.where(parallel, 1.0, direction)
        first = (lower[axis] - origin_local[axis]) / safe_direction
        second = (upper[axis] - origin_local[axis]) / safe_direction
        axis_enter = np.minimum(first, second)
        axis_exit = np.maximum(first, second)
        axis_enter[parallel] = -np.inf
        axis_exit[parallel] = np.inf
        t_enter = np.maximum(t_enter, axis_enter)
        t_exit = np.minimum(t_exit, axis_exit)
    valid &= (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit >= 0.0)
    distance = np.where(valid, np.maximum(t_enter, 0.0), np.inf)
    return distance.astype(np.float32)


def inject_virtual_opponent(
    scan: np.ndarray,
    ray_angles: np.ndarray,
    ego_pose: Pose2D,
    opponent_pose: Pose2D,
    length: float,
    width: float,
) -> np.ndarray:
    virtual_ranges = ray_rectangle_distances(
        ray_angles, ego_pose, opponent_pose, length, width
    )
    return np.minimum(np.asarray(scan, dtype=np.float32), virtual_ranges)


def relative_ego_frame(ego: Pose2D, other: Pose2D) -> tuple[float, float]:
    dx, dy = other.x - ego.x, other.y - ego.y
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def rectangle_separation(
    ego: Pose2D,
    opponent: Pose2D,
    ego_length: float,
    ego_width: float,
    opponent_length: float,
    opponent_width: float,
    inflation: float,
) -> float:
    axes = []
    for yaw in (ego.yaw, opponent.yaw):
        axes.extend(
            (
                np.asarray([math.cos(yaw), math.sin(yaw)]),
                np.asarray([-math.sin(yaw), math.cos(yaw)]),
            )
        )
    center = np.asarray([opponent.x - ego.x, opponent.y - ego.y])
    dimensions = (
        (0.5 * ego_length + inflation, 0.5 * ego_width + inflation, ego.yaw),
        (
            0.5 * opponent_length + inflation,
            0.5 * opponent_width + inflation,
            opponent.yaw,
        ),
    )
    separations = []
    for axis in axes:
        radii = []
        for half_length, half_width, yaw in dimensions:
            forward = np.asarray([math.cos(yaw), math.sin(yaw)])
            lateral = np.asarray([-math.sin(yaw), math.cos(yaw)])
            radii.append(
                half_length * abs(float(np.dot(axis, forward)))
                + half_width * abs(float(np.dot(axis, lateral)))
            )
        separations.append(abs(float(np.dot(center, axis))) - sum(radii))
    return float(max(separations))


def footprint_lidar_margin(
    scan: np.ndarray,
    ray_angles: np.ndarray,
    vehicle_length: float,
    vehicle_width: float,
    buffer: float,
) -> float:
    angles = np.asarray(ray_angles, dtype=np.float64)
    cosine = np.abs(np.cos(angles))
    sine = np.abs(np.sin(angles))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_extent = np.where(cosine > 1e-8, 0.5 * vehicle_length / cosine, np.inf)
        y_extent = np.where(sine > 1e-8, 0.5 * vehicle_width / sine, np.inf)
    extent = np.minimum(x_extent, y_extent)
    return float(np.min(np.asarray(scan) - extent) - buffer)


def compensate_scan_for_physical_footprint(
    scan: np.ndarray,
    ray_angles: np.ndarray,
    *,
    trained_length: float,
    trained_width: float,
    physical_length: float,
    physical_width: float,
) -> np.ndarray:
    """Make a larger physical chassis look conservatively closer to obstacles.

    The critics were trained with ``trained_*`` dimensions. Subtracting the
    per-ray footprint difference preserves the clearance that the critic would
    have observed if the physical vehicle had the training footprint. This is
    deliberately one-sided: a smaller physical chassis never makes obstacles
    appear farther away than they did during training.
    """
    dimensions = (trained_length, trained_width, physical_length, physical_width)
    if min(map(float, dimensions)) <= 0.0:
        raise ValueError("Trained and physical vehicle dimensions must be positive")
    values = np.asarray(scan, dtype=np.float32)
    angles = np.asarray(ray_angles, dtype=np.float64)
    if values.shape != angles.shape:
        raise ValueError("Scan and ray angles must have identical shapes")

    def extent(length: float, width: float) -> np.ndarray:
        cosine = np.abs(np.cos(angles))
        sine = np.abs(np.sin(angles))
        with np.errstate(divide="ignore", invalid="ignore"):
            longitudinal = np.where(
                cosine > 1e-8, 0.5 * float(length) / cosine, np.inf
            )
            lateral = np.where(
                sine > 1e-8, 0.5 * float(width) / sine, np.inf
            )
        return np.minimum(longitudinal, lateral)

    extra = np.maximum(
        0.0,
        extent(physical_length, physical_width)
        - extent(trained_length, trained_width),
    )
    return np.maximum(0.0, values - extra).astype(np.float32)


class ObservationBuilder:
    def __init__(self, contract: dict):
        environment = contract["environment"]
        observation = contract["observation"]
        interaction = contract["interaction"]
        composed = contract["composed_safety"]
        self.lidar_beams = int(observation["lidar_beams"])
        self.max_scan_range = float(observation["max_scan_range"])
        self.speed_scale = float(observation["speed_scale"])
        self.lateral_speed_scale = float(observation["lateral_speed_scale"])
        self.yaw_rate_scale = float(observation["yaw_rate_scale"])
        self.steering_scale = max(
            abs(float(environment["steering_min"])),
            abs(float(environment["steering_max"])),
        )
        self.command_speed_scale = max(
            abs(float(environment["speed_min"])), abs(float(environment["speed_max"]))
        )
        self.radius = float(interaction["radius_m"])
        self.max_opponents = int(interaction["max_observed_opponents"])
        self.p1_pass_buffer = float(
            composed.get("p1_pass_buffer_m", composed["pass_buffer_m"])
        )
        self.p2_pass_buffer = float(composed["pass_buffer_m"])
        self.lead_feature_scale = float(composed["lead_feature_scale_m"])
        self.lead_value_scale = float(composed["lead_value_scale_m"])
        self.value_feature_scale = float(composed["value_feature_scale"])
        self.epsilon_enter = float(composed["epsilon_enter"])
        self.postpass_mode = str(composed["postpass_mode"])
        self.formulation = str(composed["p2_formulation"])

    @property
    def p1_size(self) -> int:
        return self.lidar_beams + 5 + 7 * self.max_opponents + 1

    @property
    def p2_size(self) -> int:
        return self.p1_size + 1

    def p1(
        self,
        scan: np.ndarray,
        ego: VehicleState,
        opponent: VehicleState,
        previous_action: np.ndarray,
        ego_progress: float,
        opponent_progress: float,
    ) -> tuple[np.ndarray, float]:
        lidar = np.clip(np.asarray(scan), 0.0, self.max_scan_range)
        if lidar.shape != (self.lidar_beams,):
            raise ValueError(f"Expected {self.lidar_beams} LiDAR beams, got {lidar.shape}")
        action = np.asarray(previous_action, dtype=np.float32).reshape(2)
        dynamics = np.asarray(
            [
                np.clip(ego.longitudinal_speed / self.speed_scale, -1.0, 1.0),
                np.clip(ego.lateral_speed / self.lateral_speed_scale, -1.0, 1.0),
                np.clip(ego.yaw_rate / self.yaw_rate_scale, -1.0, 1.0),
                np.clip(action[0] / self.steering_scale, -1.0, 1.0),
                np.clip(action[1] / self.command_speed_scale, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        longitudinal, lateral = relative_ego_frame(ego.pose, opponent.pose)
        distance = math.hypot(opponent.pose.x - ego.pose.x, opponent.pose.y - ego.pose.y)
        relative_heading = opponent.pose.yaw - ego.pose.yaw
        interaction = np.zeros(7 * self.max_opponents, dtype=np.float32)
        interaction[:7] = np.asarray(
            [
                1.0,
                np.clip(longitudinal / self.radius, -1.0, 1.0),
                np.clip(lateral / self.radius, -1.0, 1.0),
                math.cos(relative_heading),
                math.sin(relative_heading),
                np.clip(
                    (opponent.longitudinal_speed - ego.longitudinal_speed) / 6.0,
                    -1.0,
                    1.0,
                ),
                np.clip(distance / self.radius, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        lead_margin = ego_progress - opponent_progress - self.p1_pass_buffer
        lead_feature = np.asarray(
            [np.clip(lead_margin / max(self.lead_feature_scale, 1e-6), -1.0, 1.0)],
            dtype=np.float32,
        )
        result = np.concatenate(
            (lidar.astype(np.float32) / self.max_scan_range, dynamics, interaction, lead_feature)
        ).astype(np.float32)
        if result.shape != (self.p1_size,):
            raise RuntimeError("P1 observation dimension contract was violated")
        return result, float(lead_margin)

    def p2(
        self,
        p1_observation: np.ndarray,
        p1_value: float,
        ego_progress: float,
        opponent_progress: float,
    ) -> tuple[np.ndarray, float]:
        task_lead = ego_progress - opponent_progress - self.p2_pass_buffer
        normalized_lead = float(
            np.clip(task_lead / max(self.lead_value_scale, 1e-6), -1.0, 1.0)
        )
        if self.formulation == "viability_constrained_avoid":
            target = normalized_lead
        elif self.formulation == "composed_target":
            viability = p1_value - self.epsilon_enter
            target = (
                min(viability, normalized_lead)
                if self.postpass_mode == "safety_only"
                else viability
            )
        else:
            raise ValueError(f"Unsupported P2 formulation: {self.formulation}")
        feature = np.asarray(
            [np.clip(target / max(self.value_feature_scale, 1e-6), -1.0, 1.0)],
            dtype=np.float32,
        )
        result = np.concatenate((np.asarray(p1_observation, np.float32), feature))
        if result.shape != (self.p2_size,):
            raise RuntimeError("P2 observation dimension contract was violated")
        return result.astype(np.float32), float(task_lead)


class SafetyModel(Protocol):
    observation_size: int

    def safe_action(self, observation: np.ndarray) -> np.ndarray: ...
    def twin_q_values(self, observation: np.ndarray, actions: np.ndarray) -> np.ndarray: ...
    def value(self, observation: np.ndarray) -> float: ...


@dataclass
class FilterResult:
    action: np.ndarray
    intervened: bool
    feasible: bool
    fallback: bool
    q_nominal: float
    q_safe: float
    required_q: float
    alpha: float
    critic_disagreement: float


class QCBFLineSearch:
    def __init__(
        self,
        model: SafetyModel,
        action_low,
        action_high,
        *,
        gamma: float,
        threshold: float,
        points: int,
        minimum_alpha: float,
        disagreement_threshold: float,
    ):
        self.model = model
        self.low = np.asarray(action_low, dtype=np.float32)
        self.high = np.asarray(action_high, dtype=np.float32)
        self.gamma = float(gamma)
        self.threshold = float(threshold)
        self.points = int(points)
        self.minimum_alpha = float(minimum_alpha)
        self.disagreement_threshold = float(disagreement_threshold)
        if self.points < 2:
            raise ValueError("Q-CBF line search requires at least two points")

    def filter(self, observation: np.ndarray, nominal_action: np.ndarray) -> FilterResult:
        nominal = np.clip(np.asarray(nominal_action, np.float32), self.low, self.high)
        safe = np.clip(self.model.safe_action(observation), self.low, self.high)
        q_safe_twins = self.model.twin_q_values(observation, safe.reshape(1, 2))[0]
        q_safe = float(np.min(q_safe_twins))
        required = self.gamma * q_safe + (1.0 - self.gamma) * self.threshold
        q_nominal_twins = self.model.twin_q_values(observation, nominal.reshape(1, 2))[0]
        q_nominal = float(np.min(q_nominal_twins))
        if q_nominal >= required:
            return FilterResult(nominal, False, True, False, q_nominal, q_safe, required, 0.0, float(np.ptp(q_nominal_twins)))
        alphas = np.linspace(0.0, 1.0, self.points, dtype=np.float32)
        candidates = np.clip((1.0 - alphas[:, None]) * nominal + alphas[:, None] * safe, self.low, self.high)
        twins = self.model.twin_q_values(observation, candidates)
        values = np.min(twins, axis=1)
        disagreements = np.ptp(twins, axis=1)
        feasible = (values >= required) & (disagreements <= self.disagreement_threshold) & (alphas >= self.minimum_alpha)
        indices = np.flatnonzero(feasible)
        if indices.size:
            index = int(indices[0])
            return FilterResult(candidates[index], True, True, False, q_nominal, q_safe, required, float(alphas[index]), float(disagreements[index]))
        # Match composed deployment: infeasibility forces the active learned actor.
        return FilterResult(safe, True, False, True, q_nominal, q_safe, required, 1.0, float(np.ptp(q_safe_twins)))


@dataclass
class SupervisorResult:
    action: np.ndarray
    phase: str
    handoff: bool
    emergency: bool
    filter_result: FilterResult


class ComposedSupervisor:
    def __init__(self, p1_filter, p2_filter, *, epsilon_enter, epsilon_emergency, confirm_steps):
        self.p1_filter = p1_filter
        self.p2_filter = p2_filter
        self.epsilon_enter = float(epsilon_enter)
        self.epsilon_emergency = float(epsilon_emergency)
        self.confirm_steps = int(confirm_steps)
        self.reset()

    def reset(self, phase: str = "p2_reach_avoid") -> None:
        if phase not in ("p2_reach_avoid", "p1_stay_safe"):
            raise ValueError(f"Unsupported composed phase: {phase!r}")
        self.phase = phase
        self.confirm_count = 0

    def rearm_p2(self) -> None:
        self.reset("p2_reach_avoid")

    def step(self, p1_observation, p2_observation, nominal, p1_value, lead_margin, g_safe):
        handoff = False
        if self.phase == "p2_reach_avoid":
            eligible = p1_value >= self.epsilon_enter and lead_margin >= 0.0 and g_safe >= 0.0
            self.confirm_count = self.confirm_count + 1 if eligible else 0
            if self.confirm_count >= self.confirm_steps:
                self.phase = "p1_stay_safe"
                handoff = True
        active_filter = self.p1_filter if self.phase == "p1_stay_safe" else self.p2_filter
        active_observation = p1_observation if self.phase == "p1_stay_safe" else p2_observation
        result = active_filter.filter(active_observation, nominal)
        # Match the current Gym deployment contract. If P2 is infeasible while
        # geometric safety is already violated, recover with the dedicated P1
        # actor rather than accelerating with the P2 fallback.
        if (
            self.phase == "p2_reach_avoid"
            and not result.feasible
            and g_safe < 0.0
        ):
            result.action = np.clip(
                self.p1_filter.model.safe_action(p1_observation),
                self.p1_filter.low,
                self.p1_filter.high,
            )
            result.intervened = True
            result.fallback = True
        emergency = self.phase == "p1_stay_safe" and p1_value < self.epsilon_emergency
        if emergency:
            result.action = np.clip(
                self.p1_filter.model.safe_action(p1_observation),
                self.p1_filter.low,
                self.p1_filter.high,
            )
            result.fallback = True
        return SupervisorResult(np.asarray(result.action, np.float32), self.phase, handoff, emergency, result)


class MPPIController:
    """NumPy MPPI matching the training nominal-controller contract."""

    def __init__(self, centerline: Centerline, config: dict, environment: dict, seed: int = 7):
        self.centerline = centerline
        self.horizon = int(config["horizon"])
        self.samples = int(config["samples"])
        self.dt = float(config["dt"])
        self.temperature = float(config["temperature"])
        self.target_speed = float(config["target_speed"])
        self.steer_sigma = float(config["steering_sigma"])
        self.speed_sigma = float(config["speed_sigma"])
        self.track_half_width = float(config["track_half_width"])
        self.steer_limit = max(abs(float(environment["steering_min"])), abs(float(environment["steering_max"])))
        # Training MPPI uses controller defaults independently of environment
        # action bounds; Q-CBF clips the returned nominal action afterward.
        self.speed_min = float(config.get("speed_min", 0.0))
        self.speed_max = float(config.get("speed_max", 6.0))
        self.wheelbase = float(config.get("wheelbase", 0.3302))
        self.rng = np.random.default_rng(seed)
        self.steering_plan = np.zeros(self.horizon, dtype=np.float64)
        self.speed_plan = np.full(self.horizon, self.target_speed, dtype=np.float64)
        self.previous_action = np.asarray([0.0, self.target_speed], dtype=np.float64)

    def reset(self) -> None:
        self.steering_plan.fill(0.0)
        self.speed_plan.fill(self.target_speed)
        self.previous_action[:] = [0.0, self.target_speed]

    def plan(self, ego: VehicleState) -> np.ndarray:
        steering = np.clip(self.steering_plan[None, :] + self.rng.normal(0.0, self.steer_sigma, (self.samples, self.horizon)), -self.steer_limit, self.steer_limit)
        speed_commands = np.clip(self.speed_plan[None, :] + self.rng.normal(0.0, self.speed_sigma, (self.samples, self.horizon)), self.speed_min, self.speed_max)
        steering[0] = self.steering_plan
        speed_commands[0] = self.speed_plan
        costs = self._costs(ego, steering, speed_commands)
        logits = np.clip(-(costs - np.min(costs)) / max(self.temperature, 1e-6), -700.0, 0.0)
        weights = np.exp(logits)
        if not np.isfinite(weights).all() or float(np.sum(weights)) <= 1e-12:
            best = int(np.argmin(costs))
            self.steering_plan = steering[best].copy()
            self.speed_plan = speed_commands[best].copy()
        else:
            weights /= np.sum(weights)
            self.steering_plan = weights @ steering
            self.speed_plan = weights @ speed_commands
        action = np.asarray([self.steering_plan[0], self.speed_plan[0]], np.float32)
        self.previous_action = action.astype(np.float64)
        self.steering_plan[:-1] = self.steering_plan[1:]
        self.speed_plan[:-1] = self.speed_plan[1:]
        if self.horizon > 1:
            self.steering_plan[-1] = self.steering_plan[-2]
            self.speed_plan[-1] = self.speed_plan[-2]
        return action

    def _costs(self, ego, steering, speed_commands):
        count = steering.shape[0]
        x = np.full(count, ego.pose.x)
        y = np.full(count, ego.pose.y)
        yaw = np.full(count, ego.pose.yaw)
        speed = np.full(count, max(0.0, ego.longitudinal_speed))
        previous_steer = np.full(count, self.previous_action[0])
        previous_speed = np.full(count, self.previous_action[1])
        costs = np.zeros(count)
        last_distance = np.zeros(count)
        nearest = self.centerline.nearest_index(ego.pose)
        indices = (nearest + np.arange(-20, 141)) % self.centerline.xy.shape[0]
        local_xy = self.centerline.xy[indices]
        following = np.roll(self.centerline.xy, -1, axis=0)
        headings = np.arctan2(following[:, 1] - self.centerline.xy[:, 1], following[:, 0] - self.centerline.xy[:, 0])[indices]
        for step in range(self.horizon):
            command = speed_commands[:, step]
            acceleration = np.clip((command - speed) / 0.35, -5.0, 3.0)
            speed = np.clip(speed + acceleration * self.dt, self.speed_min, self.speed_max)
            yaw_rate = speed * np.tan(steering[:, step]) / self.wheelbase
            yaw = wrap_angle(yaw + yaw_rate * self.dt)
            x += speed * np.cos(yaw) * self.dt
            y += speed * np.sin(yaw) * self.dt
            dx = x[:, None] - local_xy[None, :, 0]
            dy = y[:, None] - local_xy[None, :, 1]
            distances_sq = dx * dx + dy * dy
            local_index = np.argmin(distances_sq, axis=1)
            rows = np.arange(count)
            distance = np.sqrt(distances_sq[rows, local_index])
            heading_error = wrap_angle(yaw - headings[local_index])
            steer_delta = steering[:, step] - previous_steer
            speed_delta = command - previous_speed
            excess = np.maximum(0.0, distance - self.track_half_width)
            lateral_accel = speed * speed * np.tan(steering[:, step]) / self.wheelbase
            costs += (
                10.0 * distance * distance
                + 2.0 * (1.0 - np.cos(heading_error))
                + (speed - self.target_speed) ** 2
                + 0.08 * steering[:, step] ** 2
                + 0.35 * steer_delta * steer_delta
                + 0.05 * speed_delta * speed_delta
                + 400.0 * excess * excess
                + 0.02 * lateral_accel * lateral_accel
            )
            previous_steer = steering[:, step]
            previous_speed = command
            last_distance = distance
        return costs + 20.0 * last_distance * last_distance
