# F1TENTH real-ego / virtual-opponent deployment

This ROS 2 package runs the compact composed controller with a physical ego car
and a map-frame virtual opponent. The opponent can come from the included
centerline publisher or from another visualization/application that publishes
`nav_msgs/msg/Odometry`.

The controller reconstructs the trained observation, adds the virtual vehicle's
rectangle to the policy LiDAR rays, runs the nominal MPPI controller, applies the
active P2/P1 Q-CBF filter, and publishes `AckermannDriveStamped`.

## Safety defaults

- Both deployment nodes are disabled in `bringup_virtual_safety_launch.py` by default.
- The controller starts disabled and in shadow mode.
- Shadow mode publishes only `/safety_controller/shadow_drive`.
- Live mode publishes `/drive`, which is the low-priority navigation input to
  `ackermann_mux`; joystick teleoperation retains priority 100.
- Missing or older-than-150-ms scan, ego odometry, or opponent odometry causes a
  zero command in live mode.
- Initial speed is capped at 0.5 m/s.

These software checks do not replace a physical emergency stop.

## Selected deployment model

The package installs the seed-18 positive-Q hybrid bundle
`real_track_compact_seed18_qpositive_hybrid300_20260909` and the hardware launch
uses it by default. It combines the actor from the original successful seed-18
compact-track model with the Q-certified critic from the seed-18 300k-step
checkpoint. In the 10-lap seed-18 replay, the initial nominal and safe Q values
were +0.1353 and +0.1593, and the run completed four overtakes with no ego or
opponent collision in 125.76 seconds.

Initial Q was nonnegative in all five held-out evaluations, but only four of
the five completed safely; held-out seed 10018 collided with a vehicle at
1.47 seconds. The bundle therefore remains shadow-mode-only pending physical
validation. Retain the shadow-mode and 0.5 m/s commissioning limits below and
validate recorded physical inputs before allowing live commands.

The bundle contains TorchScript actors/critics, the resolved configuration,
centerline, source hashes, and dimensions. TorchScript lets the Foxy runtime
load the networks without installing Stable-Baselines3 or the training code.
The onboard Python environment still needs a compatible PyTorch build.

To reproduce the installed bundle on the training machine:

```bash
cd /home/gongkai/Research/f1tenth_safe

PYTHONPATH=f1tenth_system/f1tenth_safety_deployment \
f1tenth_safety_rl_gym/.venv/bin/python -m \
  f1tenth_safety_deployment.export_models \
  --workspace-root "$PWD" \
  --p1-model f1tenth_safety_rl_gym/outputs/real_track_compact_p1_detect1m_pass05_seed18_20260907/final_model.zip \
  --p2-model f1tenth_safety_rl_gym/outputs/real_track_compact_seed18_qpositive_hybrid300_20260909/p2_hybrid_model.zip \
  --config f1tenth_safety_rl_gym/configs/real_track_compact_ras_detect1m_pass05_10lap_1m.yaml \
  --output /tmp/real_track_compact_seed18_qpositive_hybrid300_bundle
```


The trained ego footprint is 0.29 x 0.155 m. The measured physical ego is
0.568 x 0.296 m with a 0.324 m wheelbase; the virtual opponent remains
0.29 x 0.155 m. `deployment.yaml` compensates the policy LiDAR per ray for the
ego-footprint difference and uses the physical ego dimensions for geometric
wall and opponent margins. Its MPPI wheelbase and the odometry wheelbase in
`vesc.yaml` are both 0.324 m. Recheck the chassis dimensions and calibrated
`laser_*` values whenever the physical platform or sensor mounting changes.

Copy the complete bundle to the onboard computer without modifying individual
files.

## 1. Build the ROS workspace

Place `f1tenth_system/f1tenth_stack` and
`f1tenth_system/f1tenth_safety_deployment` under the ROS workspace `src/`
directory, then:

```bash
cd /f1tenth_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Install the platform-specific PyTorch wheel separately when it is not supplied
by the Jetson image.

## 2. Start in shadow mode

Place the physical ego precisely at the configured `map_start_pose`. With
`ego_pose_is_map_frame: false`, the first `/odom` sample is aligned to that pose.
This is useful for initial tests but will drift. For multi-lap tests, publish
localized map-frame `Odometry` and set `ego_pose_is_map_frame: true` and
`ego_odom_topic` accordingly in `config/deployment.yaml`.

```bash
ros2 launch f1tenth_stack bringup_virtual_safety_launch.py \
  enable_safety_controller:=true \
  enable_virtual_opponent:=true
```

Enable computation after all topics are visible:

```bash
ros2 topic pub --once /autonomy/enable std_msgs/msg/Bool "{data: true}"
```

Inspect:

```bash
ros2 topic echo /safety_controller/status
ros2 topic echo /safety_controller/shadow_drive
ros2 topic echo /virtual_opponent/odom
```

The RViz marker is `/virtual_opponent/marker` in the `map` frame. If an external
visualizer already owns the opponent, leave `enable_virtual_opponent:=false`
and remap/configure `opponent_odom_topic` to its map-frame Odometry topic.

Reset or pause the included opponent with:

```bash
ros2 service call /virtual_opponent/reset std_srvs/srv/Trigger "{}"
ros2 service call /virtual_opponent/set_enabled std_srvs/srv/SetBool "{data: false}"
```

## 3. Enable low-speed live output only after bag replay validation

Record `/scan`, ego/opponent Odometry, shadow commands, and controller status.
Verify 33-Hz operation, input ages, map alignment, virtual LiDAR injection, and
finite Q values by replaying the bag with the wheels off the ground.

Use the supplied live commissioning profile only after the shadow checks pass.
It enables `/drive`, retains `start_enabled: false`, caps the ego at 0.5 m/s,
and lowers the open-loop opponent to 0.35 m/s:

```bash
ros2 launch f1tenth_stack bringup_virtual_safety_launch.py \
  enable_safety_controller:=true \
  enable_virtual_opponent:=true \
  safety_deployment_config:=$(ros2 pkg prefix f1tenth_safety_deployment)/share/f1tenth_safety_deployment/config/deployment_live_low_speed.yaml
```

Confirm the joystick/E-stop can stop the car, then explicitly enable autonomy:

```bash
ros2 topic pub --once /autonomy/enable std_msgs/msg/Bool "{data: true}"
```

Increase both vehicles toward the trained 1.5 m/s opponent and 3.0 m/s ego
limits only after clean bag replay, wheels-up testing, low-speed driving, and
measurement of the chassis, LiDAR transform, wheelbase, steering calibration,
and speed calibration. Publish a false enable message to command zero speed:

```bash
ros2 topic pub --once /autonomy/enable std_msgs/msg/Bool "{data: false}"
```

## Known contract limitation

Training computes the static wall margin from the map distance transform. This
deployment node currently computes the wall handoff margin from the physical
LiDAR after transforming it to the model origin. The policy observation and
Q-CBF critics are unchanged, but the handoff gate is an approximation. Validate
it in shadow mode before physical activation.
