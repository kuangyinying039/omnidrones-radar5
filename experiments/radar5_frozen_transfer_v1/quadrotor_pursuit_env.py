"""3-D pursuit game with velocity/yaw-rate commands and continuous noisy observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pursuit_evasion_env import PursuitEvasionEnv
    from pursuit_evasion_3d_env import Pursuit3DConfig, PursuitEvasion3DEnv
    from pursuit_kinematics import (
        quaternion_from_yaw,
        quaternion_to_rotation,
        yaw_from_quaternion,
    )
except ImportError:
    from .pursuit_evasion_env import PursuitEvasionEnv
    from .pursuit_evasion_3d_env import Pursuit3DConfig, PursuitEvasion3DEnv
    from .pursuit_kinematics import (
        quaternion_from_yaw,
        quaternion_to_rotation,
        yaw_from_quaternion,
    )


@dataclass
class QuadrotorPursuitConfig(Pursuit3DConfig):
    task_mode: str = "pursuit_quadrotor_3d"
    action_mode: str = "world_velocity_xyz_and_yaw_rate_to_autopilot"
    scenario_version: int = 3
    policy_observation_version: int = 2
    execution_reward_version: int = 4
    target_speed: float = 1.82  # 1.3 times pursuer horizontal speed; simulation assumption
    target_vertical_speed: float = 1.04
    initial_distance_min: float = 6.0
    initial_distance_max: float = 10.0
    formation_spacing: float = 2.2
    evader_response_time: float = 0.35
    evader_horizontal_acceleration: float = 2.5
    evader_vertical_acceleration: float = 2.0
    evader_prediction_steps: int = 6
    pursuit_target_observable: bool = False
    evader_policy: str = "occlusion"
    handoff_initial_track: bool = False
    game_position_noise_std: float = 0.10
    evader_position_noise_std: float = 0.10
    # The policy sends velocity and yaw-rate setpoints at 5 Hz.
    decision_dt: float = 0.2
    max_horizontal_velocity: float = 1.4
    max_vertical_velocity: float = 0.8
    max_reference_yaw_rate: float = 1.4
    quadrotor_gravity: float = 9.81
    quadrotor_clearance: float = 0.45
    controller_rejection_penalty: float = 0.20
    reference_smoothness_weight: float = 0.002
    safety_buffer: float = 0.10
    safety_correction_weight: float = 0.05
    boundary_proximity_weight: float = 0.03
    peer_proximity_weight: float = 0.03
    boundary_reward_safe_distance: float = 0.75
    reward_gamma: float = 0.995
    approach_distance_scale: float = 12.0
    nearest_approach_weight: float = 4.0
    capture_mode: str = "single_distance"
    target_diameter: float = 0.5  # simulation equivalent, not a hardware measurement
    capture_required_uavs: int = 1
    capture_hold_steps: int = 1
    individual_approach_weight: float = 2.0
    target_proximity_weight: float = 0.0
    encirclement_progress_weight: float = 2.0
    visibility_progress_weight: float = 1.5
    encirclement_distance_scale: float = 6.0
    encirclement_height_scale: float = 2.0
    obstacle_proximity_weight: float = 0.03
    obstacle_reward_safe_distance: float = 1.5
    obstacle_reward_log_scale: float = 0.1
    # Lidar pursuit: estimation and messaging remain diagnostics, not rewards.
    information_gain_reward: float = 0.0
    uncertain_track_penalty: float = 0.0
    communication_cost_per_kb: float = 0.0
    velocity_response_time: float = 0.35
    attitude_response_time: float = 0.20
    yaw_response_time: float = 0.20
    max_horizontal_acceleration: float = 2.5
    max_vertical_acceleration: float = 2.0
    max_tilt_deg: float = 30.0
    max_body_rate: float = 2.0
    flight_controller_substeps: int = 4
    lidar_model: str = "mid360"
    lidar_min_range: float = 0.2
    lidar_target_detection_range: float = 12.0
    lidar_horizontal_fov_deg: float = 360.0
    lidar_vertical_min_deg: float = -7.0
    lidar_vertical_max_deg: float = 52.0
    lidar_scan_hz: float = 10.0
    lidar_detection_probability: float = 0.9
    lidar_position_noise_std: float = 0.08
    lidar_noise_per_meter: float = 0.005
    lidar_mount_xyz: tuple = (0.0, 0.0, 0.0)
    lidar_mount_rpy_deg: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.execution_reward_version != 4:
            raise ValueError("Use execution_reward_version=4 for lidar visibility shaping; recollect demonstrations")
        if not 0 < self.reward_gamma <= 1 or min(self.approach_distance_scale, self.boundary_reward_safe_distance, self.safety_buffer) <= 0:
            raise ValueError("Invalid shaping discount, distance scale, or safety buffer")
        if self.target_proximity_weight != 0:
            raise ValueError("v3 uses terminal-aware potential shaping, not a per-step proximity bonus")
        if min(self.safety_correction_weight, self.boundary_proximity_weight, self.peer_proximity_weight,
               self.controller_rejection_penalty, self.obstacle_proximity_weight,
               self.individual_approach_weight, self.nearest_approach_weight, self.encirclement_progress_weight,
               self.visibility_progress_weight) < 0:
            raise ValueError("Reward weights must be nonnegative")
        if min(self.encirclement_distance_scale, self.encirclement_height_scale,
               self.obstacle_reward_safe_distance, self.obstacle_reward_log_scale) <= 0:
            raise ValueError("Reward distance scales must be positive")
        if not 0 <= self.target_proximity_weight < self.time_penalty:
            raise ValueError("Proximity reward must be below time cost to discourage lingering")
        if not np.isfinite(self.target_diameter) or self.target_diameter <= 0:
            raise ValueError("target_diameter must be finite and positive")
        self.capture_radius = 2.0 * self.target_diameter
        if self.capture_hold_steps != 1:
            raise ValueError("Distance capture succeeds immediately; capture_hold_steps must be 1")
        if self.policy_observation_version != 2:
            raise ValueError("Only pursuit policy_observation_version=2 is supported")
        if self.scenario_version != 3:
            raise ValueError("Scenario version must be 3; same-side triangle + lidar starts need recollection")
        if not 0 < self.initial_distance_min < self.initial_distance_max:
            raise ValueError("Invalid initial distance interval")
        if min(self.evader_response_time, self.evader_horizontal_acceleration, self.evader_vertical_acceleration) <= 0 or self.evader_prediction_steps < 1:
            raise ValueError("Invalid evader dynamics/prediction parameters")
        if self.target_speed <= self.max_horizontal_velocity or self.target_vertical_speed <= self.max_vertical_velocity:
            raise ValueError("Evader speed limits must exceed pursuer speed limits")
        if self.task_mode != "pursuit_quadrotor_3d" or self.action_mode != "world_velocity_xyz_and_yaw_rate_to_autopilot":
            raise ValueError("Incompatible pursuit control contract; recollect demonstrations with the current velocity/MID-360 environment")
        if min(self.game_position_noise_std, self.evader_position_noise_std) <= 0:
            raise ValueError("Game observation noise must be positive")
        if not 0 <= self.lidar_detection_probability <= 1:
            raise ValueError("lidar_detection_probability must be in [0, 1]")
        if not 0 <= self.lidar_min_range < self.lidar_target_detection_range:
            raise ValueError("Invalid lidar detection range")
        if not -90 <= self.lidar_vertical_min_deg < self.lidar_vertical_max_deg <= 90:
            raise ValueError("Invalid lidar vertical field of view")
        if not 0 < self.lidar_horizontal_fov_deg <= 360:
            raise ValueError("Invalid lidar horizontal field of view")
        if min(self.velocity_response_time, self.attitude_response_time, self.yaw_response_time, self.lidar_scan_hz,
               self.max_horizontal_acceleration, self.max_vertical_acceleration, self.max_body_rate) <= 0:
            raise ValueError("Controller time constants, limits and scan rate must be positive")
        if not 0 < self.max_tilt_deg < 80 or self.flight_controller_substeps < 1:
            raise ValueError("Invalid tilt limit or controller substeps")
        if min(self.lidar_position_noise_std, self.lidar_noise_per_meter) < 0:
            raise ValueError("Lidar noise must be nonnegative")
        self.lidar_mount_xyz = tuple(self.lidar_mount_xyz)
        self.lidar_mount_rpy_deg = tuple(self.lidar_mount_rpy_deg)
        if len(self.lidar_mount_xyz) != 3 or len(self.lidar_mount_rpy_deg) != 3:
            raise ValueError("Lidar mounting pose must have three position and three angle coordinates")
        if min(self.decision_dt, self.max_horizontal_velocity, self.max_vertical_velocity,
               self.max_reference_yaw_rate, self.quadrotor_gravity) <= 0:
            raise ValueError("Decision period, speeds, yaw-rate limit and gravity must be positive")
        if self.n_targets != 1 or self.capture_mode != "single_distance" or self.capture_required_uavs != 1:
            raise ValueError("Distance pursuit requires one evader and any one active UAV to capture it")
        if self.evader_policy not in ("repulsive", "occlusion"):
            raise ValueError("Pursuit evader_policy must be repulsive or occlusion")


class QuadrotorPursuitEnv(PursuitEvasion3DEnv):
    """HGAT/MAPPO references executed by the configured flight-control model."""

    def __init__(self, cfg: QuadrotorPursuitConfig | None = None, **kwargs):
        cfg = cfg or QuadrotorPursuitConfig()
        self._reset_lidar(cfg)
        super().__init__(cfg, **kwargs)
        self.cfg: QuadrotorPursuitConfig
        self.quadrotor_states = np.zeros((cfg.n_uavs, 13), dtype=float)
        self.previous_reference_actions = np.zeros((cfg.n_uavs, 4), dtype=float)
        self._initialize_quadrotor_states()

    def _initialize_quadrotor_states(self) -> None:
        c = self.cfg
        if not hasattr(self, "quadrotor_states") or self.quadrotor_states.shape != (c.n_uavs, 13):
            self.quadrotor_states = np.zeros((c.n_uavs, 13), dtype=float)
        for i in range(c.n_uavs):
            if hasattr(self, "initial_yaws"):
                yaw = float(self.initial_yaws[i])
            else:
                yaw = float(self.headings[i]) * np.pi / 4.0
            self.quadrotor_states[i, :3] = [*self.positions[i], self.altitudes[i]]
            self.quadrotor_states[i, 3:6] = 0.0
            self.quadrotor_states[i, 6:10] = quaternion_from_yaw(yaw)
            self.quadrotor_states[i, 10:13] = 0.0
        self.previous_reference_actions = np.zeros((c.n_uavs, 4), dtype=float)
        self._resetting_quad_state = False

    def reset(self) -> dict:
        self._reset_lidar(self.cfg)
        obs = super().reset()
        if hasattr(self, "quadrotor_states"):
            self._initialize_quadrotor_states()
            if hasattr(self, "_initialize_3d_tracking_state"):
                self._initialize_3d_tracking_state()
            self._lidar_last_scan.clear()
            self._lidar_mask_scan = -1
            self._lidar_detections[:] = False
            obs = self.observe_search()
            self._lidar_opening_scan = False
        return obs

    def _reset_lidar(self, cfg):
        self._lidar_rng = np.random.default_rng(cfg.seed + 901273)
        self._game_rng = np.random.default_rng(cfg.seed + 901274)
        self._evader_sensor_rng = np.random.default_rng(cfg.seed + 901275)
        self._game_measurement_steps = {}
        self._evader_previous_observation = None
        self._evader_estimated_velocity = np.zeros((cfg.n_uavs, 3))
        self._resetting_quad_state = True
        self._sensor_time = 0.0
        self._lidar_last_scan = {}
        self._lidar_mask_scan = -1
        self._lidar_detections = np.zeros((cfg.n_uavs, cfg.n_targets), dtype=bool)
        self._lidar_opening_scan = True
        self._visible_steps = 0

    def direct_visibility_mask(self):
        if self.cfg.pursuit_target_observable:
            return np.broadcast_to((~self.disabled_uavs)[:, None], (self.cfg.n_uavs, self.cfg.n_targets)).copy()
        try:
            from pursuit_lidar import lidar_visibility
        except ImportError:
            from .pursuit_lidar import lidar_visibility
        return lidar_visibility(self)

    def _measurement_visibility_mask(self):
        if self.cfg.pursuit_target_observable:
            return self.direct_visibility_mask()
        try:
            from pursuit_lidar import lidar_visibility
        except ImportError:
            from .pursuit_lidar import lidar_visibility
        scan = int(np.floor(self._sensor_time*self.cfg.lidar_scan_hz + 1e-9))
        if scan != self._lidar_mask_scan:
            self._lidar_detections[:] = False
            self._lidar_mask_scan = scan
        return lidar_visibility(self)

    def _target_measurement(self, agent, target_id):
        cfg = self.cfg
        if cfg.pursuit_target_observable:
            key = (agent, target_id)
            if self._game_measurement_steps.get(key) == self.t:
                return None, 0.0
            self._game_measurement_steps[key] = self.t
            truth = np.r_[self.dynamic_targets[target_id], self.target_altitudes[target_id]]
            sigma = cfg.game_position_noise_std
            return truth + self._game_rng.normal(0., sigma, 3), sigma*sigma
        scan = int(np.floor(self._sensor_time * cfg.lidar_scan_hz + 1e-9))
        key = (agent, target_id)
        if self._lidar_last_scan.get(key) == scan:
            return None, 0.0
        self._lidar_last_scan[key] = scan
        skip_dropout = bool(self._lidar_opening_scan) and cfg.lidar_detection_probability > 0
        if skip_dropout:
            try:
                from pursuit_lidar import lidar_visibility
            except ImportError:
                from .pursuit_lidar import lidar_visibility
            skip_dropout = bool(lidar_visibility(self)[agent, target_id])
        if (not skip_dropout) and self._lidar_rng.random() >= cfg.lidar_detection_probability:
            return None, 0.0
        try:
            from pursuit_lidar import sensor_pose
        except ImportError:
            from .pursuit_lidar import sensor_pose
        origin, _ = sensor_pose(self, agent)
        truth = np.r_[self.dynamic_targets[target_id], self.target_altitudes[target_id]]
        sigma = cfg.lidar_position_noise_std + cfg.lidar_noise_per_meter*np.linalg.norm(truth-origin)
        self._lidar_detections[agent, target_id] = True
        return truth + self._lidar_rng.normal(0.0, sigma, 3), float(sigma*sigma)

    def _initialize_handoff_triangle_formation(self):
        super()._initialize_handoff_triangle_formation()
        self.initial_layout = "one_side_triangle"
        target = np.r_[self.dynamic_targets[0], self.target_altitudes[0]]
        self.initial_distances = np.linalg.norm(
            np.column_stack([self.positions, self.altitudes]) - target, axis=1
        ).tolist()

    def _game_escape_direction(self, target_index):
        try:
            from pursuit_scenarios import predictive_escape
        except ImportError:
            from .pursuit_scenarios import predictive_escape
        return predictive_escape(self, target_index)

    def _advance_game_evader(self, target_index, direction, dt):
        try:
            from pursuit_scenarios import advance_evader
        except ImportError:
            from .pursuit_scenarios import advance_evader
        advance_evader(self, target_index, direction, dt)

    def continuous_action_dim(self) -> int:
        return 4

    def _safety_filtered_actions(self, actions):
        """The parent advances sensing/targets using hover placeholders only.

        Its legacy 2-D DWA/ORCA can replace hover by a grid displacement even
        when two UAVs are safely separated in altitude. That would corrupt
        positions AFTER the continuous controller integrated the flight state.
        All pursuit motion and safety are handled once, in step_joint.
        """
        actions = np.asarray(actions, dtype=int)
        if actions.shape != (self.cfg.n_uavs,) or np.any(actions != 8):
            raise ValueError('Use continuous step_joint; parent actions must be hover placeholders')
        return actions.copy()

    def obs_dim(self) -> int:
        return super().obs_dim() + 10 + 7*self.cfg.n_uavs + 5*max(self.cfg.building_state_capacity, self.cfg.building_count)

    def graph_features(self):
        """Pursuit actor features from own state, messages and local noisy tracks.

        Do not reuse the search feature builder's true-target distance/status
        or map-scanning summaries for a noisy-position pursuit experiment.
        """
        c = self.cfg
        features = np.zeros((c.n_uavs, 32))
        for i in range(c.n_uavs):
            features[i, :2] = self.positions[i]/c.grid_size
            features[i, 2:4] = [np.cos(self.headings[i]*np.pi/4), np.sin(self.headings[i]*np.pi/4)]
            if hasattr(self, 'disabled_uavs'):
                peers = self.policy_peer_mask(i)
                peers[i] = False
                if np.any(peers):
                    features[i, 4:6] = np.mean(self.positions[peers]-self.positions[i], axis=0)/c.grid_size
                features[i, 6] = np.mean(peers)
                features[i, 19:21] = [not self.disconnected_uavs[i], self.disabled_uavs[i]]
            features[i, 14] = self.altitudes[i]/c.max_altitude
            # Previously unused pursuit channels; available to flat and GAT actors.
            features[i, 21] = max(0.0, 1.0-self.t/max(c.search_steps, 1))
            if hasattr(self, 'track_memory'):
                track = self.track_memory[i][0]
                # Parent reset briefly builds planar tracks before rebuilding
                # the 3-D filter. Do not read vz from that temporary 4-D state.
                if track.initialized and len(track.mean) >= 6:
                    features[i, 7] = (track.mean[2]-self.altitudes[i])/c.max_altitude
                    features[i, 8:10] = track.mean[3:5]/max(c.target_speed, 1e-6)
                    features[i, 15] = track.mean[5]/max(c.target_vertical_speed, 1e-6)
                    relative = track.mean[:2]-self.positions[i]
                    distance = np.linalg.norm(relative)
                    features[i, 10:12] = relative/c.grid_size
                    features[i, 12] = min(distance/c.grid_size, 1.)
                    features[i, 13] = 1.
                    features[i, 17:19] = [distance <= c.tracking_radius, 1.]
                    try:
                        from pursuit_lidar import lidar_sees_point
                    except ImportError:
                        from .pursuit_lidar import lidar_sees_point
                    features[i, 16] = float(lidar_sees_point(self, i, track.mean[:3]))
        return features

    def observe_search(self) -> dict:
        """Replace the completed parent graph with policy-causal 3-D entities."""
        try:
            from pursuit_graph_encoder import pursuit_graph_observation
        except ImportError:
            from .pursuit_graph_encoder import pursuit_graph_observation
        obs = super().observe_search()
        obs["hetero_graph"] = pursuit_graph_observation(self)
        return obs

    def agent_observation_vectors(self) -> np.ndarray:
        base = super().agent_observation_vectors()
        if not hasattr(self, "quadrotor_states"):
            return np.concatenate([base, np.zeros((self.cfg.n_uavs, self.obs_dim()-base.shape[1]))], axis=1)
        velocity_scale = max(self.cfg.max_horizontal_velocity, self.cfg.max_vertical_velocity, 1e-6)
        rate_scale = max(self.cfg.max_reference_yaw_rate, 1e-6)
        rigid_body = np.concatenate(
            [
                self.quadrotor_states[:, 3:6] / velocity_scale,
                self.quadrotor_states[:, 6:10],
                self.quadrotor_states[:, 10:13] / rate_scale,
            ],
            axis=1,
        )
        # The same known map and communicated teammate states used by baselines.
        capacity = max(self.cfg.building_state_capacity, self.cfg.building_count)
        context = np.zeros((self.cfg.n_uavs, 7*self.cfg.n_uavs+5*capacity))
        for i in range(self.cfg.n_uavs):
            peers = np.zeros((self.cfg.n_uavs, 7))
            mask = self.policy_peer_mask(i)
            peers[mask, :3] = (self.quadrotor_states[mask, :3]-self.quadrotor_states[i, :3])/self.cfg.grid_size
            peers[mask, 3:6] = self.quadrotor_states[mask, 3:6]/velocity_scale
            peers[mask, 6] = 1.
            context[i, :7*self.cfg.n_uavs] = peers.ravel()
            buildings = np.zeros((capacity, 5))
            for j, (rect, height) in enumerate(zip(self.buildings, self.building_heights)):
                buildings[j] = np.r_[rect, height]/self.cfg.grid_size
            context[i, 7*self.cfg.n_uavs:] = buildings.ravel()
        return np.concatenate([base, rigid_body, context], axis=1).astype(np.float32)

    def policy_peer_mask(self, agent):
        mask = np.asarray(self.last_graph[agent] > 0, dtype=bool) if hasattr(self, 'last_graph') else np.zeros(self.cfg.n_uavs, dtype=bool)
        mask = mask & ~self.disabled_uavs
        mask[agent] = True
        return mask

    def state_vector(self) -> np.ndarray:
        base = super().state_vector()
        if not hasattr(self, "quadrotor_states"):
            return np.concatenate([base, np.zeros(self.cfg.n_uavs * 13)])
        scale = np.array(
            [
                self.cfg.grid_size,
                self.cfg.grid_size,
                self.cfg.max_altitude,
                self.cfg.max_horizontal_velocity,
                self.cfg.max_horizontal_velocity,
                self.cfg.max_vertical_velocity,
                1.0,
                1.0,
                1.0,
                1.0,
                self.cfg.max_reference_yaw_rate,
                self.cfg.max_reference_yaw_rate,
                self.cfg.max_reference_yaw_rate,
            ]
        )
        return np.concatenate([base, (self.quadrotor_states / scale).ravel()])

    def state_dim(self) -> int:
        return super().state_dim() + self.cfg.n_uavs * 13

    def _capture_geometry(self) -> tuple[bool, int, float]:
        """Any active pursuer within twice the target diameter, in world 3-D."""
        target = np.array([*self.dynamic_targets[0], self.target_altitudes[0]])
        distances = np.linalg.norm(self.quadrotor_states[:, :3] - target, axis=1)
        count = int(np.count_nonzero((distances <= self.cfg.capture_radius) & ~self.disabled_uavs))
        return count > 0, count, 0.0

    def minimum_capture_gap(self) -> float:
        """Distance remaining outside the nearest capture sphere (metres)."""
        target = np.array([*self.dynamic_targets[0], self.target_altitudes[0]])
        distances = np.linalg.norm(self.quadrotor_states[:, :3] - target, axis=1)
        active = ~self.disabled_uavs
        return float(max(np.min(distances[active])-self.cfg.capture_radius, 0.0)) if np.any(active) else float(self.cfg.grid_size)

    def step_joint(self, actions: np.ndarray) -> dict:
        try:
            from pursuit_rewards import geometry_features, shaped_rewards, obstacle_cost, clearance_costs, visibility_shaping
            from pursuit_lidar import visibility_metrics
            from pursuit_safety import safe_velocity_step, paths_conflict
        except ImportError:
            from .pursuit_rewards import geometry_features, shaped_rewards, obstacle_cost, clearance_costs, visibility_shaping
            from .pursuit_lidar import visibility_metrics
            from .pursuit_safety import safe_velocity_step, paths_conflict
        actions = np.asarray(actions, dtype=float)
        active_before = ~self.disabled_uavs.copy()
        previous_geometry = geometry_features(self.quadrotor_states[:, :3],
            [*self.dynamic_targets[0], self.target_altitudes[0]], active_before, self.cfg)
        previous_visibility = visibility_metrics(self)
        if actions.shape != (self.cfg.n_uavs, 4):
            raise ValueError(f"Expected continuous actions {(self.cfg.n_uavs, 4)}, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Velocity/yaw-rate actions must be finite")
        actions = np.clip(actions, -1.0, 1.0)
        c = self.cfg
        dt = float(getattr(self, "current_decision_dt", c.decision_dt))
        self._sensor_time += dt
        previous_states = self.quadrotor_states.copy()
        desired_velocities = np.column_stack(
            [
                actions[:, 0] * c.max_horizontal_velocity,
                actions[:, 1] * c.max_horizontal_velocity,
                actions[:, 2] * c.max_vertical_velocity,
            ]
        )
        desired_yaw_rates = actions[:, 3] * c.max_reference_yaw_rate
        horizontal = np.linalg.norm(desired_velocities[:, :2], axis=1, keepdims=True)
        desired_velocities[:, :2] *= np.minimum(1.0, c.max_horizontal_velocity/np.maximum(horizontal, 1e-12))
        feasible = active_before.copy()
        executed_velocities = np.zeros_like(desired_velocities)
        candidates = previous_states.copy()
        paths, reasons = {}, [[] for _ in range(c.n_uavs)]
        for i in range(c.n_uavs):
            if self.disabled_uavs[i]:
                continue
            candidates[i], executed_velocities[i], reasons[i], emergency, paths[i] = safe_velocity_step(
                self, i, desired_velocities[i], desired_yaw_rates[i], dt)
            feasible[i] = not emergency
        # Plan from the same pre-step team state, then validate relative sweeps.
        # Recheck after a stop because another UAV may have planned to cross its
        # vacated position. At most n_uavs passes can add new emergency stops.
        active_ids = list(paths)
        for _ in range(c.n_uavs):
            newly_stopped = False
            for index, i in enumerate(active_ids):
                for j in active_ids[index+1:]:
                    if not paths_conflict(paths[i], paths[j], c.uav_collision_radius):
                        continue
                    for agent in (i, j):
                        newly_stopped |= bool(np.any(paths[agent] != previous_states[agent, :3]))
                        feasible[agent] = False
                        candidates[agent] = previous_states[agent]
                        candidates[agent, 3:6] = 0.
                        candidates[agent, 10:13] = 0.
                        paths[agent][:] = previous_states[agent, :3]
                        executed_velocities[agent] = 0.
                        reasons[agent] = sorted(set(reasons[agent]) | {'peer'})
            if not newly_stopped:
                break
        self.quadrotor_states[:] = candidates
        scales = np.array([c.max_horizontal_velocity, c.max_horizontal_velocity, c.max_vertical_velocity])
        corrections = np.linalg.norm((desired_velocities-executed_velocities)/scales, axis=1)
        corrected = active_before & ((corrections > 1e-6) | ~feasible)
        interventions = int(np.count_nonzero(corrected))
        active_count = max(int(np.count_nonzero(active_before)), 1)

        self.positions = self.quadrotor_states[:, :2].copy()
        self.altitudes = self.quadrotor_states[:, 2].copy()
        self.vertical_velocities = self.quadrotor_states[:, 5].copy()
        yaw = np.array([yaw_from_quaternion(q) for q in self.quadrotor_states[:, 6:10]])
        self.headings = np.mod(np.rint(yaw / (np.pi / 4.0)).astype(int), 8)
        # Execute only environment sensing/communication/target dynamics; UAV
        # displacement has already been generated by the nonlinear model.
        result = PursuitEvasionEnv.step_joint(
            self, np.full(c.n_uavs, 8, dtype=int)
        )
        smoothness = c.reference_smoothness_weight * float(
            np.mean(np.sum((actions - self.previous_reference_actions) ** 2, axis=1))
        )
        infeasible_cost = c.controller_rejection_penalty * float(np.count_nonzero(active_before & ~feasible))/active_count
        correction_cost = c.safety_correction_weight * float(np.sum(np.minimum(corrections[active_before], 1.)))/active_count
        result["reward"] = float(result["reward"] - smoothness - infeasible_cost - correction_cost)
        result["continuous_actions"] = actions.tolist()
        result["desired_velocities"] = desired_velocities.tolist()
        result["executed_velocity_references"] = executed_velocities.tolist()
        result["safety_correction_rate"] = interventions/active_count
        result["safety_correction_magnitude"] = float(np.sum(corrections[active_before]))/active_count
        result["safety_reasons"] = reasons
        result["emergency_stop_rate"] = float(np.count_nonzero(active_before & ~feasible))/active_count
        result["desired_yaw_rates"] = desired_yaw_rates.tolist()
        result["quadrotor_states"] = self.quadrotor_states.tolist()
        result["initial_layout"] = self.initial_layout
        result["initial_distances"] = self.initial_distances
        result["evader_safety_interventions"] = getattr(self, "evader_safety_interventions", 0)
        result["execution_mode"] = "velocity_yaw_rate"
        result["sensor_mode"] = "noisy_game_observation" if c.pursuit_target_observable else "lidar"
        result["controller_feasible_rate"] = float(np.count_nonzero(feasible))/active_count
        result["body_angular_rates"] = self.quadrotor_states[:, 10:13].tolist()
        result["target_observation_ratio"] = float(np.mean(self.direct_visibility_mask()))
        if not c.pursuit_target_observable:
            result["lidar_detection_ratio"] = float(np.mean(self._lidar_detections))
        visible = self.direct_visibility_mask()
        result["target_visibility_rate"] = float(np.mean(np.any(visible, axis=0)))
        result["target_visible_uav_fraction"] = float(np.mean(visible))
        self._visible_steps += int(np.any(visible))
        result["target_visibility_uptime"] = self._visible_steps / max(self.t, 1)
        result["continuous_safety_interventions"] = int(interventions)
        result["safety_intervention_rate"] = max(
            float(result.get("safety_intervention_rate", 0.0)),
            interventions / max(c.n_uavs, 1),
        )
        result["step_path_length"] = float(
            np.sum(np.linalg.norm(self.quadrotor_states[:, :3] - previous_states[:, :3], axis=1))
        )
        result["control_smoothness_cost"] = smoothness
        current_capture_gap = self.minimum_capture_gap()
        current_geometry = geometry_features(self.quadrotor_states[:, :3],
            [*self.dynamic_targets[0], self.target_altitudes[0]], ~self.disabled_uavs, c)
        shaping, individual_progress = shaped_rewards(previous_geometry, current_geometry,
            active_before, ~self.disabled_uavs, c,
            terminal=bool(result["capture_success"] or self.t >= c.search_steps or result.get("terminated", False)))
        shaping["obstacle_proximity"] = obstacle_cost(self.quadrotor_states[:, :3],
            active_before, self.buildings, self.building_heights, c)
        shaping.update(clearance_costs(self.quadrotor_states[:, :3], active_before, c,
                                      np.column_stack([self.obstacles, self.obstacle_altitudes])))
        current_visibility = visibility_metrics(self)
        terminal = bool(result["capture_success"] or self.t >= c.search_steps or result.get("terminated", False))
        shaping.update(visibility_shaping(
            (previous_visibility["team_visible"], previous_visibility["uav_visibility_ratio"]),
            (current_visibility["team_visible"], current_visibility["uav_visibility_ratio"]),
            c, terminal=terminal,
        ))
        result["reward"] = float(result["reward"] + sum(shaping.values()))
        result["individual_approach_progress"] = individual_progress.tolist()
        result["encirclement_score"] = current_geometry[2]
        result["minimum_capture_gap"] = current_capture_gap
        result["team_visible"] = current_visibility["team_visible"]
        result["uav_visibility_ratio"] = current_visibility["uav_visibility_ratio"]
        result["building_occlusion_ratio"] = current_visibility["building_occlusion_ratio"]
        result["n_uavs_seeing_target"] = current_visibility["n_uavs_seeing_target"]
        result["reward_components"].update(shaping)
        result["reward_components"]["reference_smoothness"] = -smoothness
        result["reward_components"]["controller_rejection"] = -infeasible_cost
        result["reward_components"]["safety_correction"] = -correction_cost
        self.previous_reference_actions = actions.copy()
        # The parent already produced the post-action, post-capture observation.
        # Only diagnostic rewards changed since then; avoid rebuilding its graph.
        result.pop("mpc_feasible_rate", None)
        # The task has a finite deadline with an explicit failure reward.
        # There is no continuation value beyond it (unlike a rollout cutoff).
        if self.t >= c.search_steps:
            result["terminated"] = True
            result["truncated"] = False
        return result
