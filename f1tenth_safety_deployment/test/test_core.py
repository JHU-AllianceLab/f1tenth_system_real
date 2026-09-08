import math

import numpy as np

from f1tenth_safety_deployment.core import (
    Centerline,
    ComposedSupervisor,
    FilterResult,
    ObservationBuilder,
    PairProgressTracker,
    Pose2D,
    StartPoseOdomAligner,
    VehicleState,
    compensate_scan_for_physical_footprint,
    inject_virtual_opponent,
    ray_rectangle_distances,
    resample_laser_scan,
)


def contract():
    return {
        "environment": {
            "steering_min": -0.4189,
            "steering_max": 0.4189,
            "speed_min": 0.0,
            "speed_max": 3.0,
        },
        "observation": {
            "lidar_beams": 64,
            "max_scan_range": 3.0,
            "speed_scale": 3.0,
            "lateral_speed_scale": 1.5,
            "yaw_rate_scale": 5.0,
        },
        "interaction": {"radius_m": 1.0, "max_observed_opponents": 3},
        "composed_safety": {
            "p1_pass_buffer_m": 0.5,
            "pass_buffer_m": 0.5,
            "lead_feature_scale_m": 1.0,
            "lead_value_scale_m": 1.0,
            "value_feature_scale": 1.0,
            "epsilon_enter": 0.05,
            "postpass_mode": "safety_only",
            "p2_formulation": "viability_constrained_avoid",
        },
    }


def square_centerline():
    return Centerline(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    )


def test_progress_unwraps_both_vehicles_across_finish():
    tracker = PairProgressTracker(square_centerline())
    ego, opponent = tracker.update(Pose2D(0.0, 1.0, 0.0), Pose2D(0.0, 0.0, 0.0))
    assert (ego, opponent) == (3.0, 4.0)
    ego, opponent = tracker.update(Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0))
    assert (ego, opponent) == (4.0, 5.0)


def test_progress_can_retarget_same_opponent_on_a_later_lap():
    tracker = PairProgressTracker(square_centerline())
    tracker.update(Pose2D(0.0, 1.0, 0.0), Pose2D(0.0, 0.0, 0.0))

    ego, opponent = tracker.retarget_opponent_ahead(0.5)

    assert opponent - ego == 0.5


def test_start_pose_alignment_rotates_local_odometry():
    aligner = StartPoseOdomAligner(Pose2D(10.0, 20.0, math.pi / 2.0))
    aligner.transform(Pose2D(5.0, 7.0, 0.0))
    mapped = aligner.transform(Pose2D(6.0, 7.0, 0.0))
    np.testing.assert_allclose([mapped.x, mapped.y], [10.0, 21.0], atol=1e-7)
    assert math.isclose(mapped.yaw, math.pi / 2.0)


def test_virtual_rectangle_is_injected_into_scan():
    angles = np.linspace(-0.5, 0.5, 11)
    ego = Pose2D(0.0, 0.0, 0.0)
    opponent = Pose2D(1.0, 0.0, 0.0)
    distances = ray_rectangle_distances(angles, ego, opponent, 0.2, 0.2)
    assert math.isclose(float(distances[5]), 0.9, abs_tol=1e-6)
    assert np.isinf(distances[0])
    injected = inject_virtual_opponent(
        np.full(11, 3.0, dtype=np.float32), angles, ego, opponent, 0.2, 0.2
    )
    assert math.isclose(float(injected[5]), 0.9, abs_tol=1e-6)


def test_scan_is_resampled_to_training_fov_and_origin():
    sampled, angles = resample_laser_scan(
        np.full(361, 2.0),
        -math.pi,
        2.0 * math.pi / 360.0,
        beams=64,
        field_of_view=4.7,
        max_range=3.0,
    )
    assert sampled.shape == (64,)
    np.testing.assert_allclose(sampled, 2.0, atol=1e-6)
    assert math.isclose(float(angles[0]), -2.35)
    assert math.isclose(float(angles[-1]), 2.35)


def test_larger_physical_footprint_conservatively_reduces_model_scan():
    compensated = compensate_scan_for_physical_footprint(
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([0.0, math.pi / 2.0]),
        trained_length=0.2,
        trained_width=0.1,
        physical_length=0.4,
        physical_width=0.2,
    )

    np.testing.assert_allclose(compensated, [0.9, 0.95], atol=1e-6)


def test_compact_observation_has_exact_p1_and_p2_layout():
    builder = ObservationBuilder(contract())
    ego = VehicleState(Pose2D(0.0, 0.0, 0.0), 1.5, 0.0, 0.5)
    opponent = VehicleState(Pose2D(0.75, 0.0, 0.0), 1.2)
    p1, lead = builder.p1(
        np.full(64, 1.5), ego, opponent, np.asarray([0.1, 1.0]), 3.0, 3.75
    )
    p2, task_lead = builder.p2(p1, 0.2, 3.0, 3.75)
    assert p1.shape == (91,)
    assert p2.shape == (92,)
    assert math.isclose(lead, -1.25)
    assert math.isclose(task_lead, -1.25)
    np.testing.assert_allclose(p1[:64], 0.5)
    np.testing.assert_allclose(p1[69:76], [1.0, 0.75, 0.0, 1.0, 0.0, -0.05, 0.75])
    np.testing.assert_allclose(p1[76:90], 0.0)
    assert p1[-1] == -1.0
    assert p2[-1] == -1.0


class _Model:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def safe_action(self, observation):
        return self.action.copy()


class _Filter:
    def __init__(self, action):
        self.model = _Model(action)
        self.low = np.asarray([-0.4, 0.0], dtype=np.float32)
        self.high = np.asarray([0.4, 3.0], dtype=np.float32)

    def filter(self, observation, nominal):
        return FilterResult(
            action=self.model.safe_action(observation),
            intervened=True,
            feasible=False,
            fallback=True,
            q_nominal=-1.0,
            q_safe=-0.5,
            required_q=-0.4,
            alpha=1.0,
            critic_disagreement=0.0,
        )


def test_unsafe_infeasible_p2_falls_back_to_p1_recovery_actor():
    supervisor = ComposedSupervisor(
        _Filter([0.1, 0.2]),
        _Filter([-0.1, 3.0]),
        epsilon_enter=0.05,
        epsilon_emergency=0.0,
        confirm_steps=3,
    )

    result = supervisor.step(
        np.zeros(2), np.zeros(3), np.zeros(2), 0.1, -0.5, -0.01
    )

    np.testing.assert_allclose(result.action, [0.1, 0.2])
    assert result.filter_result.fallback
