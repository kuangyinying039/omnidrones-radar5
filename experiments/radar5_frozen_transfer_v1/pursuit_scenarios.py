"""Randomized same-side triangle starts and acceleration-limited predictive evasion.

Parameters are simulation assumptions pending loaded-airframe identification.
The evader observes noisy positions, not the pursuer's future commands.
"""
import numpy as np


def initialize_formation(env):
    """Place a 3-UAV triangle on one side of the target with opening lidar LOS."""
    c = env.cfg
    rng = np.random.default_rng(c.seed + 44021)
    env.initial_layout = "same_side_triangle"
    env.evader_safety_interventions = 0
    n_uavs = c.n_uavs
    spacing = float(getattr(c, "formation_spacing", 2.2))
    sensor_horizon = float(getattr(c, "lidar_target_detection_range", c.initial_distance_max)) - 1.25
    distance_max = min(c.initial_distance_max, sensor_horizon)
    distance_min = min(c.initial_distance_min, distance_max - 0.5)
    if distance_min <= 0 or distance_min >= distance_max:
        distance_min, distance_max = max(2.5, 0.45 * sensor_horizon), max(3.0, sensor_horizon)
    vertex_radius = spacing / np.sqrt(3.0)

    try:
        from pursuit_lidar import lidar_visibility
    except ImportError:
        from .pursuit_lidar import lidar_visibility

    for _ in range(2500):
        margin = min(distance_max, c.grid_size * 0.28)
        target = np.r_[rng.uniform(margin, c.grid_size - margin, 2),
                       rng.uniform(c.min_altitude + 1.2, c.max_altitude - 1.2)]
        if env._point_inside_building_prism(target, margin=0.3):
            continue
        heading = float(rng.uniform(-np.pi, np.pi))
        radial = np.array([np.cos(heading), np.sin(heading)])
        distance = float(rng.uniform(distance_min, distance_max))
        centroid_xy = target[:2] + radial * distance
        # First vertex faces the target; the other two complete an equilateral triangle.
        positions = np.zeros((n_uavs, 3), dtype=float)
        for agent in range(n_uavs):
            angle = heading + np.pi + agent * (2.0 * np.pi / max(n_uavs, 1))
            positions[agent, :2] = centroid_xy + vertex_radius * np.array([np.cos(angle), np.sin(angle)])
            positions[agent, 2] = target[2] + rng.uniform(-0.25, 0.25)
        lower = np.array([0.5, 0.5, c.min_altitude])
        upper = np.array([c.grid_size - 0.5, c.grid_size - 0.5, c.max_altitude])
        if np.any(positions < lower) or np.any(positions > upper):
            continue
        if np.any((positions[:, :2] - target[:2]) @ radial <= 0.4):
            continue
        if any(env._point_inside_building_prism(point, margin=c.quadrotor_clearance) for point in positions):
            continue
        pair_distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        pair_distance[np.diag_indices(n_uavs)] = np.inf
        if np.min(pair_distance) < max(1.5, 2 * c.quadrotor_clearance):
            continue
        env.dynamic_targets[0], env.target_altitudes[0] = target[:2], target[2]
        env.positions, env.altitudes = positions[:, :2].copy(), positions[:, 2].copy()
        yaws = np.arctan2(target[1] - env.positions[:, 1], target[0] - env.positions[:, 0])
        env.initial_yaws = yaws
        env.headings = np.mod(np.rint(yaws / (np.pi / 4.0)).astype(int), 8)
        env.target_velocity[:] = 0.0
        env.target_vertical_velocity[:] = 0.0
        env.initial_distances = np.linalg.norm(positions - target[None, :], axis=1).tolist()
        env.target_grid = env._target_occupancy()
        if hasattr(env, "quadrotor_states") and env.quadrotor_states.shape[0] == n_uavs:
            if hasattr(env, "_initialize_quadrotor_states"):
                env._initialize_quadrotor_states()
        if not np.all(lidar_visibility(env)[:, 0]):
            continue
        return
    raise ValueError("Unable to sample a same-side triangle with lidar line-of-sight; check map and range")


def velocity_step(velocity, desired, dt, cfg):
    """Stable first-order velocity response bounded by acceleration and speed."""
    velocity, desired = np.asarray(velocity), np.asarray(desired).copy()
    desired[:2] *= min(1., cfg.target_speed/max(np.linalg.norm(desired[:2]), 1e-9))
    desired[2] = np.clip(desired[2], -cfg.target_vertical_speed, cfg.target_vertical_speed)
    change = (desired-velocity)*(1-np.exp(-dt/cfg.evader_response_time))
    change[:2] *= min(1., cfg.evader_horizontal_acceleration*dt/max(np.linalg.norm(change[:2]), 1e-9))
    change[2] = np.clip(change[2], -cfg.evader_vertical_acceleration*dt, cfg.evader_vertical_acceleration*dt)
    return velocity+change


def command_velocity(direction, cfg):
    direction = np.asarray(direction)
    horizontal = direction[:2]/max(np.linalg.norm(direction[:2]), 1e-9)
    return np.r_[horizontal*cfg.target_speed, np.clip(direction[2], -1, 1)*cfg.target_vertical_speed]


def safe_segment(env, start, end):
    c = env.cfg
    return (np.all(end >= [.2, .2, c.min_altitude])
            and np.all(end <= [c.grid_size-.2, c.grid_size-.2, c.max_altitude])
            and not env._point_inside_building_prism(end, margin=.2)
            and env.has_line_of_sight_3d(start, end))


def predictive_escape(env, target_index):
    c = env.cfg
    dt = float(getattr(env, 'current_decision_dt', c.decision_dt))
    observed = np.column_stack([env.positions, env.altitudes])
    observed = observed + env._evader_sensor_rng.normal(0, c.evader_position_noise_std, observed.shape)
    if env._evader_previous_observation is not None:
        measured = (observed-env._evader_previous_observation)/dt
        measured[:, :2] *= np.minimum(1., c.max_horizontal_velocity/np.maximum(np.linalg.norm(measured[:, :2], axis=1, keepdims=True), 1e-9))
        measured[:, 2] = np.clip(measured[:, 2], -c.max_vertical_velocity, c.max_vertical_velocity)
        env._evader_estimated_velocity = .7*env._evader_estimated_velocity + .3*measured
    env._evader_previous_observation = observed
    active = ~env.disabled_uavs
    peers, peer_velocity = observed[active], env._evader_estimated_velocity[active]
    target = np.r_[env.dynamic_targets[target_index], env.target_altitudes[target_index]]
    current_velocity = np.r_[env.target_velocity[target_index], env.target_vertical_velocity[target_index]]
    best, best_score = np.zeros(3), -np.inf
    directions = [np.zeros(3)]
    directions += [np.array([np.cos(a), np.sin(a), z]) for a in np.linspace(-np.pi, np.pi, 24, endpoint=False) for z in (-.8, 0., .8)]
    for direction in directions:
        position, velocity = target.copy(), current_velocity.copy()
        desired = command_velocity(direction, c)
        clearance = np.inf
        hidden = 0
        for k in range(c.evader_prediction_steps):
            next_velocity = velocity_step(velocity, desired, dt, c)
            trial = position+.5*(velocity+next_velocity)*dt
            if not safe_segment(env, position, trial):
                break
            future_peers = peers + peer_velocity*((k+1)*dt)
            if len(peers):
                clearance = min(clearance, float(np.min(np.linalg.norm(future_peers-trial, axis=1))))
                hidden = sum(not env.has_line_of_sight_3d(peer, trial) for peer in future_peers)
            position, velocity = trial, next_velocity
        else:
            clearance = clearance if np.isfinite(clearance) else 0.
            terminal_clearance = float(np.min(np.linalg.norm(future_peers-position, axis=1))) if len(peers) else 0.
            score = (clearance + .35*terminal_clearance + float(getattr(c, "occlusion_blocked_weight", 1.2))*hidden
                     - .01*np.linalg.norm(desired-current_velocity))
            if score > best_score:
                best, best_score = direction, score
    return best


def advance_evader(env, target_index, direction, dt):
    c = env.cfg
    start = np.r_[env.dynamic_targets[target_index], env.target_altitudes[target_index]]
    velocity = np.r_[env.target_velocity[target_index], env.target_vertical_velocity[target_index]]
    next_velocity = velocity_step(velocity, command_velocity(direction, c), dt, c)
    proposed = start+.5*(velocity+next_velocity)*dt
    if not safe_segment(env, start, proposed):
        # Residual map-contact guard, not a planned instantaneous reversal/climb.
        proposed, next_velocity = start, np.zeros(3)
        env.evader_safety_interventions = getattr(env, 'evader_safety_interventions', 0)+1
    env.dynamic_targets[target_index], env.target_altitudes[target_index] = proposed[:2], proposed[2]
    env.target_velocity[target_index], env.target_vertical_velocity[target_index] = next_velocity[:2], next_velocity[2]
