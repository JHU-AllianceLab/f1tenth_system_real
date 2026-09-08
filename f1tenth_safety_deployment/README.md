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

## 1. Export a matched model pair

Wait until both P1 and P2 training finish and select a P1/P2 pair from the same
seed. Export on the Python 3.10 training machine; do not export a moving
checkpoint while it is being written.

```bash
cd /home/gongkai/Research/f1tenth_safe

PYTHONPATH=f1tenth_system/f1tenth_safety_deployment \
f1tenth_safety_rl_gym/.venv/bin/python -m \
  f1tenth_safety_deployment.export_models \
  --workspace-root "$PWD" \
  --p1-model f1tenth_safety_rl_gym/outputs/real_track_compact_p1_detect1m_pass05_seed18_20260907/final_model.zip \
  --p2-model f1tenth_safety_rl_gym/outputs/real_track_compact_ras_detect1m_pass05_10lap_seed18_20260907/deployment_eval/best_model.zip \
  --config f1tenth_safety_rl_gym/configs/real_track_compact_ras_detect1m_pass05_10lap_1m.yaml \
  --output /tmp/real_track_compact_seed18_current_bundle
```

The output contains TorchScript actors/critics, the resolved configuration,
centerline, source hashes, and dimensions. TorchScript lets the Foxy runtime
load the networks without installing Stable-Baselines3 or the training code.
The onboard Python environment still needs a compatible PyTorch build.

The selected seed-18 deployment checkpoint is the deployment-ranked 800k
checkpoint, not the final 1M checkpoint. With the current repeated-encounter
logic it completed the reference 10-lap run with four overtakes and no
collision. The controller re-arms P2 at each new encounter within 1 m.

The trained ego footprint is 0.29 x 0.155 m, while the default physical-car
envelope is 0.58 x 0.31 m. `deployment.yaml` compensates the policy LiDAR per
ray for that difference and uses the physical dimensions for geometric wall
and opponent margins. Measure the actual chassis and update
`physical_vehicle_length_m`, `physical_vehicle_width_m`, and the calibrated
`laser_*` values before live operation. The deployment MPPI wheelbase defaults
to the existing `vesc.yaml` calibration of 0.25 m; measure and update both
files together if the physical axle distance differs.

Copy the complete bundle to the onboard computer without modifying individual
files.

## 2. Build the ROS workspace

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

## 3. Start in shadow mode

Place the physical ego precisely at the configured `map_start_pose`. With
`ego_pose_is_map_frame: false`, the first `/odom` sample is aligned to that pose.
This is useful for initial tests but will drift. For multi-lap tests, publish
localized map-frame `Odometry` and set `ego_pose_is_map_frame: true` and
`ego_odom_topic` accordingly in `config/deployment.yaml`.

```bash
ros2 launch f1tenth_stack bringup_virtual_safety_launch.py \
  enable_safety_controller:=true \
  enable_virtual_opponent:=true \
  model_bundle:=/absolute/path/to/real_track_compact_seed18_current_bundle
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

## 4. Enable live output only after bag replay validation

Record `/scan`, ego/opponent Odometry, shadow commands, and controller status.
Verify 33-Hz operation, input ages, map alignment, virtual LiDAR injection, and
finite Q values by replaying the bag with the wheels off the ground.

Then set `shadow_mode: false` in `config/deployment.yaml`, rebuild, and launch
again. Keep `max_command_speed_mps: 0.5` for the first physical runs. That
commissioning cap cannot overtake the 1.5 m/s virtual opponent; first lower the
virtual opponent below 0.5 m/s, then increase both toward the trained 1.5 m/s
opponent and 3.0 m/s ego limits only after clean bag-replay and low-speed runs.
Publish a false enable message or release the hardware emergency stop to stop
autonomous operation:

```bash
ros2 topic pub --once /autonomy/enable std_msgs/msg/Bool "{data: false}"
```

## Known contract limitation

Training computes the static wall margin from the map distance transform. This
deployment node currently computes the wall handoff margin from the physical
LiDAR after transforming it to the model origin. The policy observation and
Q-CBF critics are unchanged, but the handoff gate is an approximation. Validate
it in shadow mode before physical activation.
