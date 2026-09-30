"""Reason-labelled execution checks and swept geometry for pursuit planning."""
import numpy as np

REASONS = ('boundary', 'building', 'obstacle')


def clearance_constraints(env, agent):
    """Outward normals and signed clearances; only map/own/shared peer state.

    This is a local velocity filter, not a target-seeking guidance policy.
    """
    c = env.cfg
    p = env.quadrotor_states[agent, :3]
    constraints = []
    lower = np.array([.2, .2, c.min_altitude])
    upper = np.array([c.grid_size-.2, c.grid_size-.2, c.max_altitude])
    for axis in range(3):
        normal = np.eye(3)[axis]
        constraints.extend([(normal, p[axis]-lower[axis], 'boundary', 0.),
                            (-normal, upper[axis]-p[axis], 'boundary', 0.)])
    for rect, height in zip(env.buildings, env.building_heights):
        margin = c.quadrotor_clearance
        lo = np.array([rect[0]-margin, rect[1]-margin, -1e6])
        hi = np.array([rect[2]+margin, rect[3]+margin, height+margin])
        delta = p-np.clip(p, lo, hi)
        distance = np.linalg.norm(delta)
        if distance > 1e-10:
            normal = delta/distance
        else:
            faces = np.r_[p-lo, hi-p]
            index = int(np.argmin(faces))
            normal = np.eye(3)[index % 3] * (-1 if index < 3 else 1)
            distance = -faces[index]
        constraints.append((normal, distance, 'building', 0.))
    for point in np.column_stack([env.obstacles, env.obstacle_altitudes]):
        delta = p-point
        distance = np.linalg.norm(delta)
        normal = delta/max(distance, 1e-10) if distance > 1e-10 else np.array([1., 0., 0.])
        constraints.append((normal, distance-c.obstacle_radius, 'obstacle', 0.))
    for peer in np.flatnonzero(env.policy_peer_mask(agent)):
        if peer == agent:
            continue
        delta = p-env.quadrotor_states[peer, :3]
        distance = np.linalg.norm(delta)
        normal = delta/max(distance, 1e-10) if distance > 1e-10 else np.array([1. if agent > peer else -1., 0., 0.])
        # Relative closing velocity, using the shared current peer velocity.
        offset = float(normal @ env.quadrotor_states[peer, 3:6])
        constraints.append((normal, distance-c.uav_collision_radius, 'peer', offset))
    return constraints


def project_velocity(env, agent, reference, dt):
    """Sequential half-space projection preserving tangential velocity.

    Inward speed budgets include response delay and braking distance. A
    nonlinear substep sweep below is still required: this approximation is
    not a formal control-barrier certificate for the flight model.
    """
    c = env.cfg
    velocity = np.asarray(reference, dtype=float).copy()
    constraints = clearance_constraints(env, agent)
    reasons = set()
    for _ in range(8):
        previous = velocity.copy()
        for normal, clearance, reason, offset in constraints:
            acceleration = min(c.max_horizontal_acceleration, c.max_vertical_acceleration)
            delay = c.velocity_response_time + dt
            gap = max(clearance-c.safety_buffer, 0.)
            allowed = np.sqrt((acceleration*delay)**2+2*acceleration*gap)-acceleration*delay
            # Gently leave the buffer instead of parking directly on a face.
            lower_speed = offset-allowed + min(max(c.safety_buffer-clearance, 0.)/delay, .3)
            violation = lower_speed-float(normal @ velocity)
            if violation > 1e-8:
                velocity += violation*normal
                reasons.add(reason)
        speed = np.linalg.norm(velocity[:2])
        velocity[:2] *= min(1., c.max_horizontal_velocity/max(speed, 1e-12))
        velocity[2] = np.clip(velocity[2], -c.max_vertical_velocity, c.max_vertical_velocity)
        if np.max(np.abs(velocity-previous)) < 1e-9:
            break
    return velocity, sorted(reasons)


def safe_velocity_step(env, agent, reference, yaw_rate, dt):
    """Execute a corrected reference, then sweep every integration segment.

    Emergency stop is a last-resort simulation fallback, explicitly counted;
    ordinary boundary contacts keep their feasible tangential components.
    """
    try:
        from pursuit_flight_controller import integrate_velocity_reference
    except ImportError:
        from .pursuit_flight_controller import integrate_velocity_reference
    state = env.quadrotor_states[agent]
    corrected, reasons = project_velocity(env, agent, reference, dt)
    for attempt, velocity in enumerate((corrected, np.zeros(3))):
        candidate, path = integrate_velocity_reference(state, velocity, yaw_rate, dt, env.cfg, return_path=True)
        if all(clear_segment(env, a, b) for a, b in zip(path, path[1:])):
            return candidate, velocity, reasons, bool(attempt), path
        reasons = sorted(set(reasons) | set(rejection_causes(env, candidate[:3])[0]) | {'swept_path'})
    candidate = state.copy()
    candidate[3:6] = 0.
    candidate[10:13] = 0.
    return candidate, np.zeros(3), reasons, True, np.repeat(state[None, :3], len(path), axis=0)


def paths_conflict(first, second, radius):
    """Synchronized relative-motion sweep (not just endpoint separation)."""
    relative = first-second
    for a, b in zip(relative, relative[1:]):
        delta = b-a
        fraction = np.clip(-float(a @ delta)/max(float(delta @ delta), 1e-12), 0., 1.)
        if np.linalg.norm(a+fraction*delta) <= radius:
            return True
    return False


def rejection_causes(env, position):
    c = env.cfg
    reasons, objects = [], []
    lower, upper = [.2, .2, c.min_altitude], [c.grid_size-.2, c.grid_size-.2, c.max_altitude]
    for axis in range(3):
        if position[axis] < lower[axis] or position[axis] > upper[axis]:
            objects.append(f'boundary:{axis}:{"low" if position[axis] < lower[axis] else "high"}')
    if objects:
        reasons.append('boundary')
    for index, (rect, height) in enumerate(zip(env.buildings, env.building_heights)):
        x0, y0, x1, y1 = rect
        m = c.quadrotor_clearance
        if x0-m <= position[0] <= x1+m and y0-m <= position[1] <= y1+m and position[2] <= height+m:
            if 'building' not in reasons:
                reasons.append('building')
            objects.append(f'building:{index}')
    if len(env.obstacles):
        points = np.column_stack([env.obstacles, env.obstacle_altitudes])
        for index in np.flatnonzero(np.linalg.norm(points-position, axis=1) <= c.obstacle_radius):
            if 'obstacle' not in reasons:
                reasons.append('obstacle')
            objects.append(f'obstacle:{index}')
    return reasons, objects


def segment_box(start, end, lower, upper):
    low, high = 0., 1.
    for axis in range(3):
        delta = end[axis]-start[axis]
        if abs(delta) < 1e-10:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
        else:
            a, b = (lower[axis]-start[axis])/delta, (upper[axis]-start[axis])/delta
            low, high = max(low, min(a, b)), min(high, max(a, b))
            if low > high:
                return False
    return True


def clear_segment(env, start, end, extra=0.):
    c = env.cfg
    if np.any(end < [.2, .2, c.min_altitude]) or np.any(end > [c.grid_size-.2, c.grid_size-.2, c.max_altitude]):
        return False
    m = c.quadrotor_clearance+extra
    for rect, height in zip(env.buildings, env.building_heights):
        x0, y0, x1, y1 = rect
        if segment_box(start, end, [x0-m, y0-m, -1e6], [x1+m, y1+m, height+m]):
            return False
    if len(env.obstacles):
        points = np.column_stack([env.obstacles, env.obstacle_altitudes])
        delta = end-start
        fractions = np.clip((points-start)@delta/max(delta@delta, 1e-12), 0, 1)
        if np.any(np.linalg.norm(points-(start+fractions[:, None]*delta), axis=1) <= c.obstacle_radius+extra):
            return False
    return True


class RejectionLog:
    def __init__(self, agents):
        self.counts = dict.fromkeys(REASONS, 0)
        self.streaks = np.zeros((agents, len(REASONS)), dtype=int)
        self.maximum = dict.fromkeys(REASONS, 0)
        self.object_streaks = [{} for _ in range(agents)]
        self.object_maximum = {}

    def update(self, causes, objects):
        for agent, reasons in enumerate(causes):
            for index, reason in enumerate(REASONS):
                self.streaks[agent, index] = self.streaks[agent, index]+1 if reason in reasons else 0
                self.counts[reason] += int(reason in reasons)
                self.maximum[reason] = max(self.maximum[reason], int(self.streaks[agent, index]))
            prior = self.object_streaks[agent]
            self.object_streaks[agent] = {key: prior.get(key, 0)+1 for key in objects[agent]}
            for key, value in self.object_streaks[agent].items():
                name = f'uav{agent}:{key}'
                self.object_maximum[name] = max(self.object_maximum.get(name, 0), value)

    def snapshot(self):
        return {'reason_counts': dict(self.counts), 'max_consecutive_by_reason': dict(self.maximum),
                'max_consecutive_by_object': dict(self.object_maximum)}
