# Trossen Arm Configuration Reference

Summary of all configuration terms from the [official docs](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html).

---

## Joint Modes (`trossen_arm::Mode`)

| Mode | Description |
|---|---|
| `position` | Motor tracks a target joint angle (radians) |
| `velocity` | Motor tracks a target angular velocity (rad/s) |
| `effort` | Motor applies a raw torque/force directly |
| `external_effort` | Gravity + friction compensation, plus your commanded force on top (compliant/gravity-comp mode) |
| `idle` | Holds position loosely with damping (soft brake) |

---

## Effort System

The core force/torque control pipeline.

### `effort_correction`
Scaling factor converting internal motor units to real-world units (Nm for joints, N for gripper).

```
effort_motor = effort_correction * (external_effort_desired + effort_compensation)
```

- If the arm sags under gravity in `external_effort` mode, increase this value.
- Range: [0.2, 5.0]

### `effort_compensation`
The system's estimate of torque needed to hold the arm still at its current pose. Computed from:
1. **Inverse dynamics** -- gravity/inertia torques from the kinematic model + end-effector mass
2. **Friction model** -- see below

### `external_effort`
The force/torque you command on top of compensation. In `external_effort` mode, the arm first compensates gravity+friction, then applies your additional effort. Setting to 0 = pure gravity compensation (arm floats).

### Effort Feedback
```
effort_feedback = effort_motor / effort_correction - effort_compensation
```

---

## Friction Model

Three components summed together:

```
effort_friction = constant_term
               + coulomb_coef * |effort_inverse_dynamics|
               + viscous_coef * |velocity|
```

### `friction_constant_term`
Fixed baseline friction torque, independent of speed or load. The minimum torque to overcome static friction in gears/bearings. Increase until the joint barely starts moving on its own, then back off.

### `friction_coulomb_coef`
Friction proportional to **load** (effort from inverse dynamics). Models how friction increases when the joint carries more weight. Increase if you feel more resistance when the arm is loaded vs. unloaded.

### `friction_viscous_coef`
Friction proportional to **velocity**. Models how faster motion creates more drag. Increase if faster movements feel sluggish.

### `friction_transition_velocity`
Handles the discontinuity when velocity crosses zero via linear interpolation.
- Larger value = smoother but more "stiction" feel
- Smaller value = sharper response but possible oscillation near zero

---

## Joint Characteristics

### `position_offset`
Corrects homing error per joint:
```
position_motor = position + position_offset
```
All API calls use the corrected coordinate frame.

---

## End Effector Configuration

Physical properties of the tool attached to the wrist. Critical for inverse dynamics / gravity compensation.

| Parameter | Meaning |
|---|---|
| `mass` | Total mass (kg) |
| `inertia` | 3x3 inertia matrix (9 floats) |
| `origin_xyz` | Center of mass translation from link frame |
| `origin_rpy` | Center of mass rotation (roll, pitch, yaw) |
| `palm` | Main body of end effector (excluding fingers) |
| `finger_left` / `finger_right` | Properties for each finger |
| `offset_finger_left/right` | Finger carriage center position relative to palm center when closed |
| `pitch_circle_radius` | Rack-and-pinion gear ratio for gripper mechanism |
| `t_flange_tool` | 6-element transform from wrist flange to tool tip (3 translation + 3 angle-axis rotation) |

---

## Control Loop and Limits

Internal control is a cascaded PID:

```
desired position -> [Position PID] -> velocity ref + feedforward -> [Velocity PID] -> effort -> clip -> motor
```

### Command-Side Clipping (saturates your commands)
- `position_min`, `position_max` -- joint angle limits
- `velocity_max` -- max speed
- `effort_max` -- max torque

### Feedback-Side Validation (triggers errors if exceeded)
- `position_tolerance`, `velocity_tolerance`, `effort_tolerance` -- padding beyond limits before error

### PID Parameters (`kp`, `ki`, `kd`, `imax`)
- Each motor has two PID loops: position and velocity
- `imax` = integral windup saturation limit (set to motor torque rating)

---

## Algorithm Parameters

### `singularity_threshold`
Prevents the arm from reaching singular configurations where the Jacobian loses rank:
```
error if: singularity_threshold < min|pivot_i| / max|pivot_i|
```
Computed from QR decomposition of the velocity Jacobian.

### `continuity_factors`
Scales a constraint keeping the IK solution continuous (prevents joint-space jumps between timesteps).

---

## Network Configuration

| Parameter | Meaning |
|---|---|
| `ip_method` | DHCP or manual static IP (applied at next boot) |
| `manual_ip` | Static IP address |
| `dns`, `gateway`, `subnet` | Standard network settings |
| `factory_reset_flag` | If true, all configs revert to factory defaults on next boot |
