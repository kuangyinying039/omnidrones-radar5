"""Target-detection surrogate for a body-mounted 3-D lidar, not raw point clouds."""
import math
import numpy as np

try:
    from pursuit_kinematics import quaternion_to_rotation, quaternion_from_yaw
except ImportError:
    from .pursuit_kinematics import quaternion_to_rotation, quaternion_from_yaw


def sensor_pose(env, agent):
    cfg = env.cfg
    if hasattr(env, 'quadrotor_states') and not getattr(env, '_resetting_quad_state', False):
        state = env.quadrotor_states[agent]
        position, rotation = state[:3], quaternion_to_rotation(state[6:10])
    else:
        position = np.r_[env.positions[agent], env.altitudes[agent]]
        rotation = quaternion_to_rotation(quaternion_from_yaw(float(env.headings[agent])*math.pi/4))
    roll, pitch, yaw = np.deg2rad(cfg.lidar_mount_rpy_deg)
    cx, sx, cy, sy, cz, sz = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    mount = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                      [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    return position + rotation@np.asarray(cfg.lidar_mount_xyz), rotation@mount


def lidar_visibility_report(env):
    """Geometric MID-360 visibility: range, elevation/azimuth FOV, and building LOS."""
    cfg = env.cfg
    n_uavs, n_targets = cfg.n_uavs, cfg.n_targets
    report = {
        "in_range": np.zeros((n_uavs, n_targets), dtype=bool),
        "in_fov": np.zeros((n_uavs, n_targets), dtype=bool),
        "line_of_sight": np.zeros((n_uavs, n_targets), dtype=bool),
        "visible": np.zeros((n_uavs, n_targets), dtype=bool),
        "building_occluded": np.zeros((n_uavs, n_targets), dtype=bool),
    }
    targets = np.column_stack([env.dynamic_targets, env.target_altitudes])
    disabled = np.asarray(getattr(env, "disabled_uavs", np.zeros(n_uavs, dtype=bool)), dtype=bool)
    for agent in range(n_uavs):
        if disabled[agent]:
            continue
        origin, rotation = sensor_pose(env, agent)
        local = (targets - origin) @ rotation
        ranges = np.linalg.norm(local, axis=1)
        azimuth = np.rad2deg(np.arctan2(local[:, 1], local[:, 0]))
        elevation = np.rad2deg(np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1)))
        in_range = (ranges >= cfg.lidar_min_range) & (ranges <= cfg.lidar_target_detection_range)
        in_fov = ((np.abs(azimuth) <= cfg.lidar_horizontal_fov_deg / 2)
                  & (elevation >= cfg.lidar_vertical_min_deg) & (elevation <= cfg.lidar_vertical_max_deg))
        los = np.array([env.has_line_of_sight_3d(origin, targets[target]) for target in range(n_targets)], dtype=bool)
        report["in_range"][agent] = in_range
        report["in_fov"][agent] = in_fov
        report["line_of_sight"][agent] = los
        report["visible"][agent] = in_range & in_fov & los
        report["building_occluded"][agent] = in_range & in_fov & ~los
    return report


def lidar_visibility(env):
    return lidar_visibility_report(env)["visible"]


def lidar_sees_point(env, agent, point):
    """Geometric MID-360 test for an arbitrary 3-D point in the UAV body frame."""
    cfg = env.cfg
    origin, rotation = sensor_pose(env, agent)
    local = (np.asarray(point, dtype=float) - origin) @ rotation
    range_m = float(np.linalg.norm(local))
    azimuth = np.rad2deg(np.arctan2(local[1], local[0]))
    elevation = np.rad2deg(np.arctan2(local[2], np.linalg.norm(local[:2])))
    in_range = cfg.lidar_min_range <= range_m <= cfg.lidar_target_detection_range
    in_fov = (abs(azimuth) <= cfg.lidar_horizontal_fov_deg / 2
              and cfg.lidar_vertical_min_deg <= elevation <= cfg.lidar_vertical_max_deg)
    return bool(in_range and in_fov and env.has_line_of_sight_3d(origin, np.asarray(point, dtype=float)))


def visibility_metrics(env, report=None):
    """Team and per-UAV geometric visibility used as paper metrics, not capture."""
    active = ~np.asarray(getattr(env, "disabled_uavs", np.zeros(env.cfg.n_uavs, dtype=bool)), dtype=bool)
    if env.cfg.pursuit_target_observable:
        visible = active.copy()
        occluded = np.zeros(env.cfg.n_uavs, dtype=bool)
    else:
        report = report or lidar_visibility_report(env)
        visible = report["visible"][:, 0] & active
        occluded = report["building_occluded"][:, 0] & active
    n_active = max(int(np.count_nonzero(active)), 1)
    return {
        "team_visible": bool(np.any(visible)),
        "uav_visibility_ratio": float(np.count_nonzero(visible) / n_active),
        "building_occlusion_ratio": float(np.count_nonzero(occluded) / n_active),
        "n_uavs_seeing_target": int(np.count_nonzero(visible)),
        "visible_mask": visible,
    }


def accumulate_visibility(totals, result):
    totals["team_visible"] = totals.get("team_visible", 0.0) + float(result.get("team_visible", 0.0))
    totals["uav_visibility"] = totals.get("uav_visibility", 0.0) + float(result.get("uav_visibility_ratio", 0.0))
    totals["building_occlusion"] = totals.get("building_occlusion", 0.0) + float(
        result.get("building_occlusion_ratio", 0.0)
    )
    totals["n_seeing"] = totals.get("n_seeing", 0.0) + float(result.get("n_uavs_seeing_target", 0.0))
    return totals


def finalize_visibility(totals, steps):
    n = max(int(steps), 1)
    return {
        "team_visibility_ratio": float(totals.get("team_visible", 0.0) / n),
        "uav_visibility_ratio": float(totals.get("uav_visibility", 0.0) / n),
        "building_occlusion_ratio": float(totals.get("building_occlusion", 0.0) / n),
        "mean_n_uavs_seeing_target": float(totals.get("n_seeing", 0.0) / n),
    }
