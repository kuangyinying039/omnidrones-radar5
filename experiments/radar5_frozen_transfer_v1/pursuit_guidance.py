"""Task-adapted known-map routing and velocity-response trajectory screening.

Used by FRPN/MPC guidance, not a change to the shared environment safety guard.
This is a finite-candidate controller, not a general nonlinear MPC solver.
"""
import heapq
import math
import numpy as np
from pursuit_safety import clear_segment, rejection_causes


class RoutedGuidance:
    def __init__(self):
        self.routes = {}
        self.last_fallback = {}

    def waypoint(self, env, agent, goal):
        c = env.cfg
        start = env.quadrotor_states[agent, :3]
        goal = np.clip(goal, [.3, .3, c.min_altitude+.05], [c.grid_size-.3, c.grid_size-.3, c.max_altitude-.05])
        if clear_segment(env, start, goal, .05):
            self.routes.pop(agent, None)
            return goal
        cached = self.routes.get(agent)
        if cached is not None:
            while cached and np.linalg.norm(start-cached[0]) < .5:
                cached.pop(0)
            if cached and clear_segment(env, start, cached[0], .05):
                return cached[0]
        nodes = [start.copy(), goal]
        margin = c.quadrotor_clearance+.25
        for rect, height in zip(env.buildings, env.building_heights):
            x0, y0, x1, y1 = rect
            levels = [start[2]]
            if c.min_altitude < height+margin < c.max_altitude:
                levels.append(height+margin)
            for z in levels:
                for x, y in ((x0-margin, y0-margin), (x0-margin, y1+margin),
                             (x1+margin, y0-margin), (x1+margin, y1+margin)):
                    point = np.array([x, y, z])
                    if not rejection_causes(env, point)[0]:
                        nodes.append(point)
        costs, parent, queue = {0: 0.}, {}, [(0., 0)]
        visited = set()
        while queue:
            cost, i = heapq.heappop(queue)
            if i in visited:
                continue
            visited.add(i)
            if i == 1:
                path, index = [], 1
                while index != 0:
                    path.append(nodes[index])
                    index = parent[index]
                path.reverse()
                # The final moving target waypoint is refreshed after static corners.
                self.routes[agent] = path[:-1]
                return path[0]
            for j, point in enumerate(nodes):
                if j in visited or not clear_segment(env, nodes[i], point, .05):
                    continue
                candidate = cost+np.linalg.norm(point-nodes[i])
                if candidate < costs.get(j, np.inf):
                    costs[j], parent[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        return start.copy()

    def select(self, env, agent, preferred, goal, reference_action, horizon=8):
        c = env.cfg
        start_state = env.quadrotor_states[agent]
        waypoint = self.waypoint(env, agent, goal)
        direct = waypoint-start_state[:3]
        direct /= max(np.linalg.norm(direct), 1e-9)
        candidates = [np.asarray(preferred), np.zeros(3)]
        for scale in (.35, .7, 1.):
            for angle in (-np.pi/3, 0., np.pi/3):
                x, y = direct[:2]
                direction = np.array([np.cos(angle)*x-np.sin(angle)*y,
                                      np.sin(angle)*x+np.cos(angle)*y, direct[2]])
                candidates.append(direction*c.max_horizontal_velocity*scale)
        candidates += [np.array([0., 0., sign*c.max_vertical_velocity]) for sign in (-1, 1)]
        candidates.append(-start_state[3:6])
        actions = np.array([reference_action(env, agent, velocity) for velocity in candidates])
        desired = actions[:, :3]*[c.max_horizontal_velocity, c.max_horizontal_velocity, c.max_vertical_velocity]
        desired[:, :2] *= np.minimum(1., c.max_horizontal_velocity/np.maximum(np.linalg.norm(desired[:, :2], axis=1, keepdims=True), 1e-9))
        positions, velocities = np.repeat(start_state[None, :3], len(actions), axis=0), np.repeat(start_state[None, 3:6], len(actions), axis=0)
        valid = np.ones(len(actions), dtype=bool)
        safe_steps = np.zeros(len(actions), dtype=int)
        dt = float(getattr(env, 'current_decision_dt', c.decision_dt))
        substeps = max(c.flight_controller_substeps, math.ceil(dt/(.5*min(c.velocity_response_time, c.attitude_response_time, c.yaw_response_time))))
        h = dt/substeps
        peers = env.policy_peer_mask(agent)
        peers[agent] = False
        peer_positions, peer_velocities = env.quadrotor_states[peers, :3], env.quadrotor_states[peers, 3:6]
        for step in range(horizon*substeps):
            acceleration = (desired-velocities)/c.velocity_response_time
            limit = min(c.max_horizontal_acceleration, c.quadrotor_gravity*math.tan(math.radians(c.max_tilt_deg)))
            acceleration[:, :2] *= np.minimum(1., limit/np.maximum(np.linalg.norm(acceleration[:, :2], axis=1, keepdims=True), 1e-9))
            acceleration[:, 2] = np.clip(acceleration[:, 2], -c.max_vertical_acceleration, c.max_vertical_acceleration)
            next_velocities = velocities+h*acceleration
            next_positions = positions+.5*h*(velocities+next_velocities)
            future_peers = peer_positions+peer_velocities*((step+1)*h)
            for k in np.flatnonzero(valid):
                if not clear_segment(env, positions[k], next_positions[k]):
                    valid[k] = False
                elif len(future_peers) and np.min(np.linalg.norm(future_peers-next_positions[k], axis=1)) < c.uav_collision_radius+.1:
                    valid[k] = False
                else:
                    safe_steps[k] += 1
            positions, velocities = next_positions, next_velocities
        cost = np.sum((positions-waypoint)**2, axis=1)+.08*np.sum((desired-np.asarray(preferred))**2, axis=1)
        candidates_ok = np.flatnonzero(valid)
        self.last_fallback[agent] = not bool(len(candidates_ok))
        if not len(candidates_ok):
            # No safe sequence: longest predicted safe prefix, then least residual speed.
            candidates_ok = np.flatnonzero(safe_steps == np.max(safe_steps))
            index = int(candidates_ok[np.argmin(np.linalg.norm(velocities[candidates_ok], axis=1))])
        else:
            index = int(candidates_ok[np.argmin(cost[candidates_ok])])
        return actions[index]
