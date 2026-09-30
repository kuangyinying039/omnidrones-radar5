"""Reduced-order autopilot surrogate for velocity + yaw-rate references.

World frame is z-up, body frame is x-forward/y-left/z-up. This is a training
surrogate, not a PX4 driver or a torque-level rigid-body simulation.
"""
import math
import numpy as np

try:
    from pursuit_kinematics import quaternion_to_rotation, yaw_from_quaternion
except ImportError:
    from .pursuit_kinematics import quaternion_to_rotation, yaw_from_quaternion


def _euler_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def integrate_velocity_reference(state, velocity, yaw_rate, dt, cfg, *, return_path=False):
    """Lagged, acceleration-limited velocity tracking with rate-limited tilt."""
    result = np.asarray(state, dtype=float).copy()
    path = [result[:3].copy()] if return_path else None
    target = np.asarray(velocity, dtype=float).copy()
    speed = np.linalg.norm(target[:2])
    if speed > cfg.max_horizontal_velocity:
        target[:2] *= cfg.max_horizontal_velocity/speed
    target[2] = np.clip(target[2], -cfg.max_vertical_velocity, cfg.max_vertical_velocity)
    yaw_rate = float(np.clip(yaw_rate, -cfg.max_reference_yaw_rate, cfg.max_reference_yaw_rate))
    # Keep Euler lag integration stable for calibrated short response times.
    substeps = max(cfg.flight_controller_substeps, math.ceil(float(dt) /
                   (0.5 * min(cfg.velocity_response_time, cfg.attitude_response_time, cfg.yaw_response_time))))
    step = float(dt)/substeps
    for _ in range(substeps):
        acceleration = (target-result[3:6])/cfg.velocity_response_time
        horizontal = np.linalg.norm(acceleration[:2])
        limit = min(cfg.max_horizontal_acceleration, cfg.quadrotor_gravity*math.tan(math.radians(cfg.max_tilt_deg)))
        if horizontal > limit:
            acceleration[:2] *= limit/horizontal
        acceleration[2] = np.clip(acceleration[2], -cfg.max_vertical_acceleration, cfg.max_vertical_acceleration)
        old_velocity = result[3:6].copy()
        result[3:6] += step*acceleration
        result[:3] += 0.5*step*(old_velocity+result[3:6])
        if return_path:
            path.append(result[:3].copy())
        rotation = quaternion_to_rotation(result[6:10])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
        yaw = yaw_from_quaternion(result[6:10])
        current_yaw_rate = (result[11]*math.sin(roll)+result[12]*math.cos(roll))/max(math.cos(pitch), 1e-6)
        tracked_yaw_rate = current_yaw_rate + (yaw_rate-current_yaw_rate)*step/cfg.yaw_response_time
        forward_acc = math.cos(yaw)*acceleration[0]+math.sin(yaw)*acceleration[1]
        left_acc = -math.sin(yaw)*acceleration[0]+math.cos(yaw)*acceleration[1]
        pitch_target = math.atan2(forward_acc, cfg.quadrotor_gravity)
        roll_target = -math.atan2(left_acc, cfg.quadrotor_gravity)
        roll_dot = float(np.clip((roll_target-roll)/cfg.attitude_response_time, -cfg.max_body_rate, cfg.max_body_rate))
        pitch_dot = float(np.clip((pitch_target-pitch)/cfg.attitude_response_time, -cfg.max_body_rate, cfg.max_body_rate))
        # Euler yaw rate and body z rate are not identical when tilted.
        body_rates = np.array([roll_dot-tracked_yaw_rate*math.sin(pitch),
                              pitch_dot*math.cos(roll)+tracked_yaw_rate*math.sin(roll)*math.cos(pitch),
                              -pitch_dot*math.sin(roll)+tracked_yaw_rate*math.cos(roll)*math.cos(pitch)])
        factor = min(1.0, cfg.max_body_rate/max(float(np.max(np.abs(body_rates))), 1e-12))
        result[10:13] = body_rates*factor
        result[6:10] = _euler_quaternion(roll+step*roll_dot*factor,
                                        pitch+step*pitch_dot*factor, yaw+step*tracked_yaw_rate*factor)
    return (result, np.asarray(path)) if return_path else result
