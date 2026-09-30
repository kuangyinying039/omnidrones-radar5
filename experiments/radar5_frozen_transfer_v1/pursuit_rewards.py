"""Bounded geometric rewards for the distance-capture task only."""
import numpy as np


def geometry_features(positions, target, active, cfg):
    delta = np.asarray(positions) - np.asarray(target)
    distances = np.linalg.norm(delta, axis=1)
    # Inside the capture sphere there is no further approach incentive.
    log_distance = np.log(np.maximum(distances, cfg.capture_radius) / cfg.capture_radius)
    proximity = np.where(active, 1.0 / (1.0 + log_distance), 0.0).mean()
    # Horizontal encirclement is useful only near the target and at its altitude.
    # Three vehicles cannot enclose a volume in full 3-D.
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    eligible = np.asarray(active) & (horizontal > 1e-6)
    enclosure = 0.0
    if np.count_nonzero(eligible) >= 3:
        angles = np.sort(np.arctan2(delta[eligible, 1], delta[eligible, 0]))
        max_gap = np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))
        angular = np.clip((2*np.pi-max_gap)/(2*np.pi*(1-1/len(angles))), 0, 1)
        near = np.exp(-np.mean(np.maximum(distances[eligible]-cfg.capture_radius, 0))/cfg.encirclement_distance_scale)
        altitude = np.exp(-np.mean(np.abs(delta[eligible, 2]))/cfg.encirclement_height_scale)
        enclosure = float(angular * near * altitude)
    return log_distance, float(proximity), enclosure


def obstacle_cost(positions, active, buildings, heights, cfg):
    """3-D distance to solid building boxes, less UAV clearance radius."""
    risk = np.zeros(len(positions))
    for rect, height in zip(buildings, heights):
        lower, upper = np.array([rect[0], rect[1], 0.]), np.array([rect[2], rect[3], height])
        outside = np.maximum(np.maximum(lower-positions, positions-upper), 0.)
        clearance = np.maximum(np.linalg.norm(outside, axis=1)-cfg.quadrotor_clearance, 0.)
        penetration = np.maximum(cfg.obstacle_reward_safe_distance-clearance, 0.)
        risk = np.maximum(risk, np.log1p(penetration/cfg.obstacle_reward_log_scale)/
                          np.log1p(cfg.obstacle_reward_safe_distance/cfg.obstacle_reward_log_scale))
    return -cfg.obstacle_proximity_weight * float(np.where(active, risk, 0.).mean())


def clearance_costs(positions, active, cfg, obstacles):
    """Bounded, continuous costs for boundary, spherical obstacles and peers."""
    points = np.asarray(positions)
    active = np.asarray(active, dtype=bool)
    lower = np.array([.2, .2, cfg.min_altitude])
    upper = np.array([cfg.grid_size-.2, cfg.grid_size-.2, cfg.max_altitude])
    clearance = np.minimum(points-lower, upper-points).min(axis=1)
    boundary = np.clip(1.-clearance/cfg.boundary_reward_safe_distance, 0., 1.)**2
    obstacle = np.zeros(len(points))
    for point in obstacles:
        gap = np.linalg.norm(points-point, axis=1)-cfg.obstacle_radius
        obstacle = np.maximum(obstacle, np.clip(1.-gap/cfg.obstacle_reward_safe_distance, 0., 1.)**2)
    peer = np.zeros(len(points))
    for i in np.flatnonzero(active):
        others = active.copy()
        others[i] = False
        if np.any(others):
            gap = np.min(np.linalg.norm(points[others]-points[i], axis=1))-cfg.uav_collision_radius
            peer[i] = np.clip(1.-gap/cfg.boundary_reward_safe_distance, 0., 1.)**2
    # Fixed team denominator: a dead UAV does not amplify survivors' costs.
    return {"boundary_proximity": -cfg.boundary_proximity_weight*float(np.where(active, boundary, 0.).mean()),
            "point_obstacle_proximity": -cfg.obstacle_proximity_weight*float(np.where(active, obstacle, 0.).mean()),
            "peer_proximity": -cfg.peer_proximity_weight*float(peer.mean())}


def shaped_rewards(previous, current, active_before, active_after, cfg, terminal=False):
    """Discount-consistent potentials, zero at *all* episode terminal states.

    Nearest-agent progress matches any-one-UAV capture; mean progress and
    altitude-aware enclosure encourage the other UAVs to assist. No teacher
    actions or target-truth observations are injected into the actor.
    """
    old_log, _, old_enclosure = previous
    new_log, _, enclosure = current
    old_potential = np.where(active_before, np.exp(-cfg.capture_radius*np.expm1(old_log)/cfg.approach_distance_scale), 0.)
    new_potential = np.where(active_after, np.exp(-cfg.capture_radius*np.expm1(new_log)/cfg.approach_distance_scale), 0.)
    if terminal:
        new_potential = np.zeros_like(new_potential)
        enclosure = 0.
    progress = cfg.reward_gamma*new_potential-old_potential
    return {
        "individual_approach": cfg.individual_approach_weight * float(progress.mean()),
        "nearest_approach": cfg.nearest_approach_weight * float(cfg.reward_gamma*new_potential.max()-old_potential.max()),
        "target_proximity": 0.0,
        "encirclement_progress": cfg.encirclement_progress_weight * (cfg.reward_gamma*enclosure-old_enclosure),
    }, progress


def visibility_potential(team_visible, uav_visibility_ratio):
    return float(team_visible) + 0.5 * float(uav_visibility_ratio)


def visibility_shaping(previous, current, cfg, terminal=False):
    """Potential on team/per-UAV geometric lidar visibility; zero at terminals."""
    old_potential = visibility_potential(*previous)
    new_potential = 0.0 if terminal else visibility_potential(*current)
    return {
        "visibility_progress": cfg.visibility_progress_weight * (cfg.reward_gamma * new_potential - old_potential)
    }
