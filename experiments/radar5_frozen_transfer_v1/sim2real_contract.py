"""Unit-safe boundary between the Radar5 pursuit policy and Isaac Lab 3.

The policy uses world z-up coordinates and wxyz quaternions. Isaac Lab 3
exposes xyzw quaternions. Policy actions are normalized world velocity and
yaw-rate setpoints at 5 Hz; a physical controller consumes the returned
velocity and yaw angle. This module does not read simulator ground truth for
the actor or implement a flight controller.
"""

from __future__ import annotations

import numpy as np


def isaac_xyzw_to_policy_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a batch of finite Isaac Lab xyzw quaternions to policy wxyz."""
    q = np.asarray(quaternion, dtype=np.float32)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite with shape [..., 4]")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("quaternion norm must be nonzero")
    q = q / norm
    return q[..., [3, 0, 1, 2]]


def policy_wxyz_to_isaac_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert finite Radar5 wxyz quaternions to normalized Isaac Lab xyzw."""
    q = np.asarray(quaternion, dtype=np.float32)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("quaternion must be finite with shape [..., 4]")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("quaternion norm must be nonzero")
    q = q / norm
    return q[..., [1, 2, 3, 0]]


def policy_action_to_setpoint(
    policy_action: np.ndarray,
    measured_yaw: np.ndarray,
    *,
    decision_dt: float = 0.2,
    max_horizontal_velocity: float = 1.4,
    max_vertical_velocity: float = 0.8,
    max_yaw_rate: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized ``[..., agents, 4]`` action to world m/s and yaw radians.

    Measured yaw has shape ``[..., agents]``. Recompute the target once each
    policy decision and hold it during the intervening physics steps.
    """
    action = np.asarray(policy_action, dtype=np.float32)
    yaw = np.asarray(measured_yaw, dtype=np.float32)
    if action.ndim < 2 or action.shape[-1] != 4:
        raise ValueError("policy_action must have shape [..., agents, 4]")
    if yaw.shape != action.shape[:-1]:
        raise ValueError("measured_yaw must match policy_action without its last axis")
    if not np.isfinite(action).all() or not np.isfinite(yaw).all():
        raise ValueError("policy_action and measured_yaw must be finite")
    if min(decision_dt, max_horizontal_velocity, max_vertical_velocity, max_yaw_rate) <= 0:
        raise ValueError("decision_dt and limits must be positive")
    clipped = np.clip(action, -1.0, 1.0)
    velocity = clipped[..., :3] * np.array(
        [max_horizontal_velocity, max_horizontal_velocity, max_vertical_velocity],
        dtype=np.float32,
    )
    horizontal_norm = np.linalg.norm(velocity[..., :2], axis=-1, keepdims=True)
    velocity[..., :2] *= np.minimum(
        1.0, max_horizontal_velocity / np.maximum(horizontal_norm, 1e-12)
    )
    target = yaw + clipped[..., 3] * max_yaw_rate * decision_dt
    target = np.arctan2(np.sin(target), np.cos(target))
    return velocity, target


def policy_action_to_lee_velocity_command(
    policy_action: np.ndarray,
    measured_yaw: np.ndarray,
    *,
    max_horizontal_velocity: float = 1.4,
    max_vertical_velocity: float = 0.8,
    max_yaw_rate: float = 1.4,
) -> np.ndarray:
    """Convert world-frame policy action to Lee's yaw-aligned velocity command.

    LeeVelController expects [vx, vy, vz, yaw_rate] in the vehicle frame
    (world frame rotated by minus measured yaw), not a yaw-angle target.
    The caller holds this command for one 0.2 s policy decision interval.
    """
    action = np.asarray(policy_action, dtype=np.float32)
    yaw = np.asarray(measured_yaw, dtype=np.float32)
    if action.ndim < 2 or action.shape[-1] != 4 or yaw.shape != action.shape[:-1]:
        raise ValueError("expected action [..., agents, 4] and yaw [..., agents]")
    if not np.isfinite(action).all() or not np.isfinite(yaw).all():
        raise ValueError("action and yaw must be finite")
    if min(max_horizontal_velocity, max_vertical_velocity, max_yaw_rate) <= 0:
        raise ValueError("velocity and yaw-rate limits must be positive")
    clipped = np.clip(action, -1.0, 1.0)
    world_vx = clipped[..., 0] * max_horizontal_velocity
    world_vy = clipped[..., 1] * max_horizontal_velocity
    horizontal_norm = np.hypot(world_vx, world_vy)
    horizontal_scale = np.minimum(
        1.0, max_horizontal_velocity / np.maximum(horizontal_norm, 1e-12)
    )
    world_vx *= horizontal_scale
    world_vy *= horizontal_scale
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    body_vx = cos_yaw * world_vx + sin_yaw * world_vy
    body_vy = -sin_yaw * world_vx + cos_yaw * world_vy
    return np.stack((
        body_vx,
        body_vy,
        clipped[..., 2] * max_vertical_velocity,
        clipped[..., 3] * max_yaw_rate,
    ), axis=-1).astype(np.float32)
