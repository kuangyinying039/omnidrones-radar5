"""Three-dimensional distributed pursuit and tracking extension.

The target belief is a six-state constant-velocity process
``[x, y, z, vx, vy, vz]``. Each UAV performs local 3-D measurement updates and
freshness-aware covariance-intersection over directed links. The occlusion
evader evaluates lateral, far-side, and roof candidates around building
prisms. Motion remains kinematic; attitude, thrust and rotor dynamics belong
to the subsequent nonlinear-NMPC stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pursuit_evasion_env import PursuitConfig, PursuitEvasionEnv, _unit, _unit_rows
    from target_tracker import TrackMessage, TrackState3D, covariance_intersection, fuse_track_with_message
except ImportError:
    from .pursuit_evasion_env import PursuitConfig, PursuitEvasionEnv, _unit, _unit_rows
    from .target_tracker import TrackMessage, TrackState3D, covariance_intersection, fuse_track_with_message


@dataclass
class Pursuit3DConfig(PursuitConfig):
    task_mode: str = "pursuit_3d"
    # Current paper scope: static building prisms only. Dynamic airborne
    # obstacles remain implemented for later extensions but are disabled.
    n_obstacles: int = 0
    dynamic_obstacles_enabled: bool = False
    min_altitude: float = 1.0
    max_altitude: float = 12.0
    initial_altitude: float = 5.0
    vertical_speed: float = 0.8
    target_vertical_speed: float = 0.65
    vertical_field_of_view_deg: float = 80.0
    building_min_height: float = 3.0
    building_max_height: float = 10.0
    capture_vertical_tolerance: float = 1.5
    handoff_altitude_noise: float = 0.45
    handoff_altitude_variance: float = 0.36
    vertical_measurement_noise: float = 0.10
    occlusion_candidate_margin: float = 0.8
    occlusion_lookahead_steps: float = 4.0
    # Calibrated after the formal 5-seed APF/FRPN screen: 1.8 yielded zero
    # captures even with >95% NMPC feasibility. 1.2 keeps active cover seeking
    # without making immediate LOS denial dominate every candidate decision.
    occlusion_blocked_weight: float = 1.2
    occlusion_clearance_weight: float = 0.12
    occlusion_travel_weight: float = 0.05
    occlusion_min_waypoint_dwell_steps: int = 10
    occlusion_switch_score_margin: float = 0.35
    handoff_triangle_formation: bool = True
    handoff_formation_min_distance: float = 4.0
    handoff_formation_max_distance: float = 6.0
    obstacle_vertical_speed: float = 0.35


class PursuitEvasion3DEnv(PursuitEvasionEnv):
    """3-D kinematic pursuit with 11 actions: planar 9 plus climb/descent."""

    def __init__(self, cfg: Pursuit3DConfig | None = None, **kwargs):
        cfg = cfg or Pursuit3DConfig()
        self._use_3d_tracking = False
        # These defaults are required because the base constructor calls
        # observation methods before this constructor completes.
        self.altitudes = np.full(cfg.n_uavs, cfg.initial_altitude, dtype=float)
        self.vertical_velocities = np.zeros(cfg.n_uavs, dtype=float)
        self.target_altitudes = np.full(cfg.n_targets, cfg.initial_altitude, dtype=float)
        self.target_vertical_velocity = np.zeros(cfg.n_targets, dtype=float)
        self.obstacle_altitudes = np.full(cfg.n_obstacles, cfg.initial_altitude, dtype=float)
        self.obstacle_vertical_velocity = np.zeros(cfg.n_obstacles, dtype=float)
        self.building_heights = np.zeros(cfg.building_count, dtype=float)
        super().__init__(cfg, **kwargs)
        self._reset_3d_state()

    def _reset_3d_state(self) -> None:
        c = self.cfg
        self.altitudes = np.clip(
            c.initial_altitude + self.rng.uniform(-1.0, 1.0, size=c.n_uavs),
            c.min_altitude,
            c.max_altitude,
        )
        self.vertical_velocities = np.zeros(c.n_uavs, dtype=float)
        self.target_altitudes = np.clip(
            c.initial_altitude + self.rng.uniform(-1.5, 1.5, size=c.n_targets),
            c.min_altitude,
            c.max_altitude,
        )
        self.target_vertical_velocity = self.rng.uniform(
            -c.target_vertical_speed, c.target_vertical_speed, size=c.n_targets
        )
        self.occlusion_waypoints = np.full((c.n_targets, 3), np.nan, dtype=float)
        self.occlusion_waypoint_dwell = np.zeros(c.n_targets, dtype=np.int32)
        self.building_heights = self.rng.uniform(
            c.building_min_height, c.building_max_height, size=len(self.buildings)
        )
        self.obstacle_altitudes = self.rng.uniform(
            c.min_altitude, c.max_altitude, size=len(self.obstacles)
        )
        self.obstacle_vertical_velocity = self.rng.uniform(
            -c.obstacle_vertical_speed, c.obstacle_vertical_speed, size=len(self.obstacles)
        )
        if c.handoff_triangle_formation and c.n_uavs == 3 and c.n_targets >= 1:
            # A dense building layout can leave the sampled target with no
            # feasible same-side triangle. Resample that initial target before
            # giving up; the buildings and seed remain unchanged.
            last_error = None
            for attempt in range(128):
                if attempt:
                    candidate = np.r_[
                        self.rng.uniform(0.5, c.grid_size - 0.5, size=2),
                        np.clip(c.initial_altitude + self.rng.uniform(-1.5, 1.5),
                                c.min_altitude, c.max_altitude),
                    ]
                    if self._point_inside_building_prism(candidate, margin=0.2):
                        continue
                    self.dynamic_targets[0] = candidate[:2]
                    self.target_altitudes[0] = candidate[2]
                try:
                    self._initialize_handoff_triangle_formation()
                except RuntimeError as error:
                    last_error = error
                    continue
                if attempt:
                    self.target_grid = self._target_occupancy()
                break
            else:
                raise RuntimeError(
                    "Unable to sample a visible collision-free 3-D handoff triangle"
                ) from last_error
        self._use_3d_tracking = True
        self._initialize_3d_tracking_state()

    def _initialize_handoff_triangle_formation(self) -> None:
        """Place a visible three-UAV triangle on one side of the target.

        Candidate search respects map bounds and building prisms. UAV 0 is
        selected from candidates with direct 3-D LOS and is pointed toward the
        target, guaranteeing a physically grounded initial local observation.
        """
        c = self.cfg
        target = np.array([*self.dynamic_targets[0], self.target_altitudes[0]], dtype=float)
        radii = np.linspace(c.handoff_formation_min_distance + 0.25,
                            c.handoff_formation_max_distance - 0.25, 4)
        angle_offsets = (0.0, -np.pi / 18.0, np.pi / 18.0,
                         -np.pi / 9.0, np.pi / 9.0)

        def candidates(expected_angle: float):
            for radius in radii:
                for angle_offset in angle_offsets:
                    angle = expected_angle + angle_offset
                    for dz in (0.0, 0.35, -0.35):
                        horizontal = float(np.sqrt(max(radius * radius - dz * dz, 0.0)))
                        point = target + np.array(
                            [horizontal * np.cos(angle), horizontal * np.sin(angle), dz]
                        )
                        if not (0.5 <= point[0] <= c.grid_size - 0.5
                                and 0.5 <= point[1] <= c.grid_size - 0.5
                                and c.min_altitude <= point[2] <= c.max_altitude):
                            continue
                        if self._point_inside_building_prism(point, margin=c.quadrotor_clearance if hasattr(c, "quadrotor_clearance") else 0.45):
                            continue
                        yield point

        base_angle = float(self.rng.uniform(-np.pi, np.pi))
        anchor = next(
            (point for point in candidates(base_angle) if self.has_line_of_sight_3d(point, target)),
            None,
        )
        if anchor is None:
            # Search the complete azimuth if the nominal side is obstructed.
            anchor = next(
                (
                    point
                    for angle in np.linspace(-np.pi, np.pi, 37)[:-1]
                    for point in candidates(float(angle))
                    if self.has_line_of_sight_3d(point, target)
                ),
                None,
            )
        if anchor is None:
            raise RuntimeError("Unable to construct a visible 3-D handoff formation.")
        selected = [anchor]
        anchor_angle = float(np.arctan2(anchor[1] - target[1], anchor[0] - target[0]))
        for agent, offset in ((1, -np.pi / 5.0), (2, np.pi / 5.0)):
            expected = anchor_angle + offset
            point = next(
                (
                    candidate
                    for candidate in candidates(expected)
                    if self.has_line_of_sight_3d(candidate, target)
                    and all(np.linalg.norm(candidate - other) >= 2.0 * 0.45 for other in selected)
                ),
                None,
            )
            if point is None:
                point = next(
                    (
                        candidate
                        for angle in np.linspace(expected - np.pi / 8.0, expected + np.pi / 8.0, 17)
                        for candidate in candidates(float(angle))
                        if self.has_line_of_sight_3d(candidate, target)
                        and all(np.linalg.norm(candidate - other) >= 2.0 * 0.45 for other in selected)
                    ),
                    None,
                )
            if point is None:
                raise RuntimeError("Unable to construct a collision-free 3-D handoff triangle.")
            selected.append(point)
        formation = np.asarray(selected)
        self.positions = formation[:, :2].copy()
        self.altitudes = formation[:, 2].copy()
        toward_target = target[None, :2] - self.positions
        direction_table = np.stack(
            [self.uav_step_vector(action) for action in range(8)]
        ).astype(float)
        direction_table /= np.maximum(np.linalg.norm(direction_table, axis=1, keepdims=True), 1e-9)
        normalized_target = toward_target / np.maximum(
            np.linalg.norm(toward_target, axis=1, keepdims=True), 1e-9
        )
        self.headings = np.argmax(normalized_target @ direction_table.T, axis=1).astype(int)

    def reset(self) -> dict:
        # The base constructor invokes reset before the 3-D fields are fully
        # available. In that phase, retain the base 2-D initialization and
        # replace it once construction completes.
        if not hasattr(self, "evader_policy"):
            return super().reset()
        self._use_3d_tracking = False
        # Base reset builds temporary 2-D observations before rebuilding tracks.
        # Do not let it update last episode's 6-D tracks into 3-D belief arrays.
        if hasattr(self, "target_belief_mean"):
            del self.target_belief_mean
        obs = super().reset()
        self._reset_3d_state()
        return self.observe_search()

    def _initialize_3d_tracking_state(self) -> None:
        c = self.cfg
        self.track_memory = [
            [TrackState3D.uninitialized(target_id) for target_id in range(c.n_targets)]
            for _ in range(c.n_uavs)
        ]
        self.target_belief_mean = np.zeros((c.n_uavs, c.n_targets, 3), dtype=float)
        self.target_belief_velocity = np.zeros((c.n_uavs, c.n_targets, 3), dtype=float)
        self.target_belief_covariance = np.repeat(
            np.eye(6, dtype=float)[None, None, :, :], c.n_uavs * c.n_targets, axis=0
        ).reshape(c.n_uavs, c.n_targets, 6, 6)
        self.target_belief_timestamp = np.full((c.n_uavs, c.n_targets), -1, dtype=np.int32)
        self.target_belief_source = np.full((c.n_uavs, c.n_targets), -1, dtype=np.int32)
        self.last_tracking_information_gain = 0.0
        self.last_track_messages = 0
        self.last_track_bytes = 0
        self.last_scheduled_track_ids = {}
        self._belief_updated_at = -1
        if c.handoff_initial_track:
            for target_id in range(c.n_targets):
                measurement = np.array(
                    [
                        *(
                            self.dynamic_targets[target_id]
                            + self.rng.normal(0.0, c.handoff_position_noise, size=2)
                        ),
                        self.target_altitudes[target_id]
                        + self.rng.normal(0.0, c.handoff_altitude_noise),
                    ],
                    dtype=float,
                )
                measurement[:2] = np.clip(measurement[:2], 0.0, c.grid_size - 1e-6)
                measurement[2] = np.clip(measurement[2], c.min_altitude, c.max_altitude)
                position_variance = float(
                    (2.0 * c.handoff_position_variance + c.handoff_altitude_variance) / 3.0
                )
                for uav_id in range(c.n_uavs):
                    self.track_memory[uav_id][target_id].initialize(
                        measurement,
                        timestamp=0,
                        measurement_variance=position_variance,
                        velocity_variance=c.handoff_velocity_variance,
                    )
                    self.target_belief_source[uav_id, target_id] = -2
        self._update_target_beliefs(force=True)

    def action_dim(self) -> int:
        return 11

    def obs_dim(self) -> int:
        return 43

    def action_mask(self, i: int) -> np.ndarray:
        planar = super().action_mask(i)
        mask = np.ones(11, dtype=bool)
        mask[:9] = planar
        if hasattr(self, "disabled_uavs") and self.disabled_uavs[i]:
            mask[:] = False
            mask[8] = True
            return mask
        altitude = float(self.altitudes[i])
        mask[9] = altitude < self.cfg.max_altitude - 1e-6
        mask[10] = altitude > self.cfg.min_altitude + 1e-6
        return mask

    def agent_observation_vectors(self) -> np.ndarray:
        features = self.graph_features().copy()
        # Reuse two task-summary channels for normalized own altitude and
        # vertical velocity, preserving the 32 semantic feature contract.
        features[:, 30] = self.altitudes / max(self.cfg.max_altitude, 1e-6)
        features[:, 31] = self.vertical_velocities / max(self.cfg.vertical_speed, 1e-6)
        masks = np.stack([self.action_mask(i) for i in range(self.cfg.n_uavs)]).astype(float)
        return np.concatenate([features, masks], axis=1)

    def has_line_of_sight_3d(self, start: np.ndarray, end: np.ndarray) -> bool:
        for rect, height in zip(self.buildings, self.building_heights):
            interval = _segment_rect_interval(start[:2], end[:2], rect)
            if interval is None:
                continue
            lo, hi = interval
            z_lo = float(start[2] + lo * (end[2] - start[2]))
            z_hi = float(start[2] + hi * (end[2] - start[2]))
            if min(z_lo, z_hi) <= height:
                return False
        return True

    def direct_visibility_mask(self) -> np.ndarray:
        c = self.cfg
        if c.pursuit_target_observable:
            return np.broadcast_to(
                (~self.disabled_uavs)[:, None], (c.n_uavs, c.n_targets)
            ).copy()
        pursuers = np.column_stack([self.positions.astype(float), self.altitudes])
        targets = np.column_stack([self.dynamic_targets, self.target_altitudes])
        delta = targets[None, :, :] - pursuers[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        horizontal_distance = np.linalg.norm(delta[:, :, :2], axis=2)
        heading = _unit_rows(
            np.stack([self.uav_step_vector(int(h)) for h in self.headings]).astype(float)
        )
        horizontal_direction = delta[:, :, :2] / np.maximum(horizontal_distance[:, :, None], 1e-8)
        horizontal_cos = np.sum(horizontal_direction * heading[:, None, :], axis=2)
        horizontal_ok = horizontal_cos >= np.cos(np.deg2rad(c.field_of_view_deg * 0.5))
        elevation = np.abs(np.arctan2(delta[:, :, 2], np.maximum(horizontal_distance, 1e-8)))
        vertical_ok = elevation <= np.deg2rad(c.vertical_field_of_view_deg * 0.5)
        los = np.ones_like(distance, dtype=bool)
        for i in range(c.n_uavs):
            for j in range(c.n_targets):
                los[i, j] = self.has_line_of_sight_3d(pursuers[i], targets[j])
        return (
            (~self.disabled_uavs)[:, None]
            & (distance <= c.sensor_radius)
            & horizontal_ok
            & vertical_ok
            & los
        )

    def _target_measurement(self, agent, target_id):
        c = self.cfg
        truth = np.array([*self.dynamic_targets[target_id], self.target_altitudes[target_id]], dtype=float)
        measurement = truth + np.array([
            *self.rng.normal(0.0, c.belief_measurement_noise, size=2),
            self.rng.normal(0.0, c.vertical_measurement_noise),
        ])
        variance = float((2.0*c.belief_measurement_noise**2+c.vertical_measurement_noise**2)/3.0)
        return measurement, variance

    def _measurement_visibility_mask(self):
        return self.direct_visibility_mask()

    def _update_target_beliefs(self, *, force: bool = False) -> None:
        """Run a 6-D CV-KF and freshness-aware 6-D CI on each UAV."""
        if not getattr(self, "_use_3d_tracking", False):
            return super()._update_target_beliefs(force=force)
        if not force and self._belief_updated_at == self.t:
            return
        c = self.cfg
        dt = float(getattr(self, "current_decision_dt", c.decision_dt))
        for tracks in self.track_memory:
            for track in tracks:
                track.predict(dt, c.belief_process_noise)

        visible = self._measurement_visibility_mask()
        information_gain = 0.0
        for i in range(c.n_uavs):
            for target_id in np.flatnonzero(visible[i]):
                measurement, measurement_variance = self._target_measurement(i, target_id)
                if measurement is None:
                    continue
                information_gain += self.track_memory[i][target_id].update_xyz(
                    measurement,
                    self.t,
                    measurement_variance,
                    c.belief_initial_velocity_variance,
                )
                self.target_belief_source[i, target_id] = i
                self.last_direct_detection_step[i, target_id] = self.t

        messages = {
            (source, target_id): TrackMessage.from_track(source, track)
            for source, tracks in enumerate(self.track_memory)
            for target_id, track in enumerate(tracks)
            if track.initialized
        }
        message_count = 0
        scheduled: dict[int, list[int]] = {}
        if hasattr(self, "last_graph"):
            for source in range(c.n_uavs):
                if self.disabled_uavs[source]:
                    continue
                receivers = [
                    int(receiver)
                    for receiver in np.flatnonzero(self.last_graph[:, source] > 0)
                    if not self.disabled_uavs[int(receiver)]
                ]
                priorities = []
                for target_id in range(c.n_targets):
                    if (source, target_id) not in messages or not receivers:
                        continue
                    source_message = messages[(source, target_id)]
                    score = 0.0
                    for receiver in receivers:
                        receiver_message = messages.get((receiver, target_id))
                        if receiver_message is None:
                            score = max(score, float(c.belief_max_age + 1))
                            continue
                        _, fused_covariance, _ = covariance_intersection(
                            receiver_message.mean,
                            receiver_message.covariance,
                            source_message.mean,
                            source_message.covariance,
                            grid_size=21,
                        )
                        reduction = max(
                            0.0,
                            float(np.trace(receiver_message.covariance[:3, :3]))
                            - float(np.trace(fused_covariance[:3, :3])),
                        )
                        age = min(max(self.t - receiver_message.timestamp, 0), c.belief_max_age)
                        score = max(score, (1.0 + age) * reduction)
                    priorities.append((score, target_id))
                priorities.sort(key=lambda item: (-item[0], item[1]))
                selected = [
                    target_id
                    for _, target_id in priorities[: max(0, min(c.max_tracks_per_message, len(priorities)))]
                ]
                scheduled[source] = selected
                for receiver in receivers:
                    for target_id in selected:
                        information_gain += fuse_track_with_message(
                            self.track_memory[receiver][target_id],
                            messages[(source, target_id)],
                            now=self.t if c.track_freshness_fusion_enabled else None,
                            freshness_tau=c.track_freshness_tau if c.track_freshness_fusion_enabled else None,
                            link_confidence=float(np.clip(self.last_graph[receiver, source], 0.0, 1.0)),
                            confidence_floor=c.track_confidence_floor,
                        )
                        message_count += 1
                        self.target_belief_source[receiver, target_id] = source

        self.last_tracking_information_gain = float(information_gain / max(c.n_targets, 1))
        self.last_track_messages = int(message_count)
        self.last_track_bytes = int(message_count * c.track_message_bytes)
        self.last_scheduled_track_ids = scheduled
        self._sync_track_arrays()
        self._belief_updated_at = self.t

    def _sync_track_arrays(self) -> None:
        if not getattr(self, "_use_3d_tracking", False):
            return super()._sync_track_arrays()
        for i, tracks in enumerate(self.track_memory):
            for target_id, track in enumerate(tracks):
                if not track.initialized:
                    self.target_belief_timestamp[i, target_id] = -1
                    continue
                self.target_belief_mean[i, target_id] = track.mean[:3]
                self.target_belief_velocity[i, target_id] = track.mean[3:]
                self.target_belief_covariance[i, target_id] = track.covariance
                self.target_belief_timestamp[i, target_id] = track.timestamp

    def _track_evaluation_metrics(self) -> dict[str, float]:
        if not getattr(self, "_use_3d_tracking", False):
            return super()._track_evaluation_metrics()
        position_errors, velocity_errors, nees_values, ages = [], [], [], []
        for tracks in self.track_memory:
            for target_id, track in enumerate(tracks):
                if not track.initialized:
                    continue
                truth_position = np.array(
                    [*self.dynamic_targets[target_id], self.target_altitudes[target_id]]
                )
                truth_velocity = np.array(
                    [*self.target_velocity[target_id], self.target_vertical_velocity[target_id]]
                )
                position_error = track.mean[:3] - truth_position
                velocity_error = track.mean[3:] - truth_velocity
                position_errors.append(float(np.linalg.norm(position_error)))
                velocity_errors.append(float(np.linalg.norm(velocity_error)))
                ages.append(track.age(self.t, self.cfg.belief_max_age))
                try:
                    nees_values.append(
                        float(position_error @ np.linalg.solve(track.covariance[:3, :3], position_error))
                    )
                except np.linalg.LinAlgError:
                    nees_values.append(float("nan"))
        finite_nees = np.asarray(nees_values, dtype=float)
        finite_nees = finite_nees[np.isfinite(finite_nees)]
        return {
            "track_position_rmse": float(np.sqrt(np.mean(np.square(position_errors))) if position_errors else np.nan),
            "track_velocity_rmse": float(np.sqrt(np.mean(np.square(velocity_errors))) if velocity_errors else np.nan),
            "mean_track_nees": float(np.mean(finite_nees) if len(finite_nees) else np.nan),
            # 95% chi-square interval for a three-dimensional position error.
            "track_nees_consistency_rate": float(
                np.mean((finite_nees >= 0.2158) & (finite_nees <= 9.3484))
                if len(finite_nees)
                else np.nan
            ),
            "mean_track_age": float(np.mean(ages) if ages else self.cfg.belief_max_age),
            "p95_track_age": float(np.percentile(ages, 95) if ages else self.cfg.belief_max_age),
        }

    def _move_dynamic_entities(self) -> None:
        c = self.cfg
        dt = float(getattr(self, "current_decision_dt", c.decision_dt))
        if c.dynamic_targets_enabled and not self.captured:
            for idx in range(c.n_targets):
                if c.evader_policy == "repulsive" and hasattr(self, "_game_escape_direction"):
                    direction = self._game_escape_direction(idx)
                elif c.evader_policy == "occlusion":
                    direction = self._occlusion_direction_3d(idx)
                else:
                    horizontal = self.evader_policy.action(
                        self, idx, self.evader_observation(idx)
                    )
                    active_altitudes = self.altitudes[~self.disabled_uavs]
                    dz = self.target_altitudes[idx] - float(
                        np.mean(active_altitudes) if len(active_altitudes) else c.initial_altitude
                    )
                    vertical = np.sign(dz) if abs(dz) >= 0.6 else self.rng.choice([-1.0, 1.0])
                    direction = _unit(np.array([horizontal[0], horizontal[1], 0.45 * vertical]))
                if hasattr(self, "_advance_game_evader"):
                    self._advance_game_evader(idx, direction, dt)
                    continue
                horizontal = _unit(direction[:2], self.target_velocity[idx])
                self.target_velocity[idx] = horizontal * c.target_speed
                self.target_vertical_velocity[idx] = float(
                    np.clip(direction[2] * c.target_vertical_speed, -c.target_vertical_speed, c.target_vertical_speed)
                )
                current = np.array(
                    [*self.dynamic_targets[idx], self.target_altitudes[idx]], dtype=float
                )
                proposed = current + np.array(
                    [
                        self.target_velocity[idx, 0] * dt,
                        self.target_velocity[idx, 1] * dt,
                        self.target_vertical_velocity[idx] * dt,
                    ]
                )
                proposed[:2] = np.clip(proposed[:2], 0.0, c.grid_size - 1e-6)
                proposed[2] = np.clip(proposed[2], c.min_altitude, c.max_altitude)
                if self._point_inside_building_prism(proposed, margin=0.2):
                    # Prefer climbing over the roof when feasible; otherwise
                    # reject the horizontal penetration and reverse direction.
                    roof = self._covering_building_height(proposed[:2], margin=0.2)
                    if roof is not None and roof + c.occlusion_candidate_margin < c.max_altitude:
                        proposed[:2] = current[:2]
                        proposed[2] = min(
                            c.max_altitude,
                            current[2] + c.target_vertical_speed * dt,
                        )
                        self.target_vertical_velocity[idx] = c.target_vertical_speed
                    else:
                        proposed = current
                        self.target_velocity[idx] *= -1.0
                        self.target_vertical_velocity[idx] *= -0.5
                self.dynamic_targets[idx] = proposed[:2]
                self.target_altitudes[idx] = proposed[2]
        if c.dynamic_obstacles_enabled:
            self.obstacles, self.obstacle_velocity = self._move_points(
                self.obstacles, self.obstacle_velocity
            )
            self.obstacle_altitudes += self.obstacle_vertical_velocity * dt
            low = self.obstacle_altitudes < c.min_altitude
            high = self.obstacle_altitudes > c.max_altitude
            self.obstacle_vertical_velocity[low | high] *= -1.0
            self.obstacle_altitudes = np.clip(
                self.obstacle_altitudes, c.min_altitude, c.max_altitude
            )
        self.target_grid = self._target_occupancy()

    def _positions_3d(self) -> np.ndarray:
        return np.column_stack([self.positions.astype(float), self.altitudes])

    def _count_pair_conflicts(self) -> int:
        positions = self._positions_3d()
        conflicts = 0
        for i in range(self.cfg.n_uavs):
            if self.disabled_uavs[i]:
                continue
            for j in range(i):
                if not self.disabled_uavs[j] and np.linalg.norm(positions[i] - positions[j]) < self.cfg.safe_radius:
                    conflicts += 1
        return conflicts

    def _count_uav_collisions(self) -> tuple[int, set[int]]:
        positions = self._positions_3d()
        collisions = 0
        crashed: set[int] = set()
        for i in range(self.cfg.n_uavs):
            if self.disabled_uavs[i]:
                continue
            for j in range(i):
                if self.disabled_uavs[j]:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) <= self.cfg.uav_collision_radius:
                    collisions += 1
                    crashed.update((i, j))
        return collisions, crashed

    def _count_obstacle_conflicts(self) -> tuple[int, set[int]]:
        if len(self.obstacles) == 0:
            return 0, set()
        obstacles = np.column_stack([self.obstacles.astype(float), self.obstacle_altitudes])
        positions = self._positions_3d()
        crashed: set[int] = set()
        for i, position in enumerate(positions):
            if not self.disabled_uavs[i] and np.min(np.linalg.norm(obstacles - position, axis=1)) <= self.cfg.obstacle_radius:
                crashed.add(i)
        return len(crashed), crashed

    def _occlusion_direction_3d(self, target_index: int) -> np.ndarray:
        """Select a collision-free 3-D direction that maximizes prism occlusion."""
        c = self.cfg
        target = np.array(
            [*self.dynamic_targets[target_index], self.target_altitudes[target_index]], dtype=float
        )
        active = ~self.disabled_uavs
        pursuers = np.column_stack([self.positions[active].astype(float), self.altitudes[active]])
        if not len(pursuers):
            return _unit(
                np.array([*self.target_velocity[target_index], self.target_vertical_velocity[target_index]])
            )
        delta = target[None, :] - pursuers
        distances = np.linalg.norm(delta, axis=1)
        weights = 1.0 / np.maximum(distances, 0.5) ** 2
        repulsion = np.sum(delta / np.maximum(distances[:, None], 1e-8) * weights[:, None], axis=0)
        repulsion = _unit(repulsion, np.array([*self.target_velocity[target_index], 0.0]))
        candidates: list[np.ndarray] = []
        margin = c.occlusion_candidate_margin
        pursuer_center = np.mean(pursuers, axis=0)
        for rect, height in zip(self.buildings, self.building_heights):
            x0, y0, x1, y1 = rect
            center_xy = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
            cover_z = float(np.clip(min(target[2], height - margin), c.min_altitude, c.max_altitude))
            side_points = (
                np.array([x0 - margin, center_xy[1], cover_z]),
                np.array([x1 + margin, center_xy[1], cover_z]),
                np.array([center_xy[0], y0 - margin, cover_z]),
                np.array([center_xy[0], y1 + margin, cover_z]),
            )
            candidates.extend(side_points)
            # A far-side point explicitly lies behind the prism relative to
            # the pursuer centroid; a roof point allows a rational overflight
            # when lateral cover would be a dead end.
            away_xy = _unit(center_xy - pursuer_center[:2], target[:2] - pursuer_center[:2])
            radius = 0.5 * max(x1 - x0, y1 - y0) + margin
            candidates.append(np.array([*(center_xy + away_xy * radius), cover_z]))
            if height + margin <= c.max_altitude:
                candidates.append(np.array([*center_xy, height + margin]))

        lookahead = max(c.occlusion_lookahead_steps * c.target_speed, c.target_speed)
        candidates.append(target + repulsion * lookahead)
        best_score = -float("inf")
        best_direction = repulsion
        best_waypoint = target + repulsion * lookahead

        def score_waypoint(waypoint: np.ndarray) -> tuple[float, np.ndarray] | None:
            waypoint = np.asarray(waypoint, dtype=float).copy()
            waypoint[:2] = np.clip(waypoint[:2], 0.2, c.grid_size - 0.2)
            waypoint[2] = np.clip(waypoint[2], c.min_altitude, c.max_altitude)
            direction = _unit(waypoint - target, repulsion)
            trial = target + direction * lookahead
            trial[:2] = np.clip(trial[:2], 0.2, c.grid_size - 0.2)
            trial[2] = np.clip(trial[2], c.min_altitude, c.max_altitude)
            if self._point_inside_building_prism(trial, margin=0.2):
                return None
            blocked = sum(not self.has_line_of_sight_3d(pursuer, trial) for pursuer in pursuers)
            clearance = float(np.min(np.linalg.norm(pursuers - trial[None, :], axis=1)))
            travel = float(np.linalg.norm(waypoint - target))
            score = (
                c.occlusion_blocked_weight * blocked
                + c.occlusion_clearance_weight * clearance
                - c.occlusion_travel_weight * travel
                + 0.25 * float(np.dot(direction, repulsion))
            )
            return float(score), direction

        for waypoint in candidates:
            evaluated = score_waypoint(waypoint)
            if evaluated is None:
                continue
            score, direction = evaluated
            if score > best_score:
                best_score = score
                best_direction = direction
                best_waypoint = np.asarray(waypoint, dtype=float).copy()

        current_waypoint = self.occlusion_waypoints[target_index]
        current_evaluation = (
            score_waypoint(current_waypoint) if np.all(np.isfinite(current_waypoint)) else None
        )
        reached = bool(
            np.all(np.isfinite(current_waypoint))
            and np.linalg.norm(current_waypoint - target) <= 1.5 * c.occlusion_candidate_margin
        )
        may_switch = self.occlusion_waypoint_dwell[target_index] >= c.occlusion_min_waypoint_dwell_steps
        should_switch = (
            current_evaluation is None
            or reached
            or (may_switch and best_score >= current_evaluation[0] + c.occlusion_switch_score_margin)
        )
        if should_switch:
            self.occlusion_waypoints[target_index] = best_waypoint
            self.occlusion_waypoint_dwell[target_index] = 0
            chosen_direction = best_direction
        else:
            self.occlusion_waypoint_dwell[target_index] += 1
            chosen_direction = current_evaluation[1]
        return _unit(0.25 * repulsion + 0.75 * chosen_direction, repulsion)

    def _covering_building_height(self, xy: np.ndarray, margin: float = 0.0) -> float | None:
        heights = [
            float(height)
            for rect, height in zip(self.buildings, self.building_heights)
            if rect[0] - margin <= xy[0] <= rect[2] + margin
            and rect[1] - margin <= xy[1] <= rect[3] + margin
        ]
        return max(heights) if heights else None

    def _point_inside_building_prism(self, point: np.ndarray, margin: float = 0.0) -> bool:
        height = self._covering_building_height(np.asarray(point)[:2], margin=margin)
        return bool(height is not None and float(point[2]) <= height + margin)

    def _capture_geometry(self) -> tuple[bool, int, float]:
        target_xy = self.dynamic_targets[0]
        horizontal_delta = self.positions.astype(float) - target_xy[None, :]
        dz = self.altitudes - self.target_altitudes[0]
        distance_3d = np.sqrt(np.sum(horizontal_delta**2, axis=1) + dz**2)
        close = np.flatnonzero(
            (distance_3d <= self.cfg.capture_radius)
            & (np.abs(dz) <= self.cfg.capture_vertical_tolerance)
            & ~self.disabled_uavs
        )
        if len(close) < self.cfg.capture_required_uavs:
            return False, len(close), 0.0
        angles = np.sort(np.mod(np.arctan2(horizontal_delta[close, 1], horizontal_delta[close, 0]), 2 * np.pi))
        wrapped = np.r_[angles, angles[0] + 2 * np.pi]
        span = 2 * np.pi - float(np.max(np.diff(wrapped)))
        return span >= np.deg2rad(self.cfg.capture_angular_span_deg), len(close), float(np.rad2deg(span))

    def step_joint(self, search_actions: np.ndarray) -> dict:
        actions = np.asarray(search_actions, dtype=int)
        if actions.shape != (self.cfg.n_uavs,):
            raise ValueError(f"Expected {(self.cfg.n_uavs,)} actions, got {actions.shape}")
        dt = float(getattr(self, "current_decision_dt", self.cfg.decision_dt))
        self.vertical_velocities.fill(0.0)
        climb = actions == 9
        descend = actions == 10
        self.vertical_velocities[climb] = self.cfg.vertical_speed
        self.vertical_velocities[descend] = -self.cfg.vertical_speed
        self.altitudes = np.clip(
            self.altitudes + self.vertical_velocities * dt,
            self.cfg.min_altitude,
            self.cfg.max_altitude,
        )
        planar_actions = actions.copy()
        planar_actions[actions >= 9] = 8
        result = super().step_joint(planar_actions)
        result["raw_actions"] = actions.tolist()
        result["executed_actions"] = actions.tolist()
        result["mean_altitude"] = float(np.mean(self.altitudes))
        result["target_altitude"] = float(self.target_altitudes[0])
        result["obs"] = self.observe_search()
        return result

    def observe_search(self) -> dict:
        obs = super().observe_search()
        if getattr(self, "_use_3d_tracking", False):
            graph = obs["hetero_graph"]
            for i in range(self.cfg.n_uavs):
                for target_id, track in enumerate(self.track_memory[i]):
                    if target_id >= graph["track_nodes"].shape[1] or not track.initialized:
                        continue
                    node = graph["track_nodes"][i, target_id]
                    relative_z = self.target_belief_mean[i, target_id, 2] - self.altitudes[i]
                    node[4] = min(
                        float(
                            np.linalg.norm(
                                self.target_belief_mean[i, target_id]
                                - np.array([*self.positions[i], self.altitudes[i]])
                            )
                        )
                        / self.cfg.grid_size,
                        1.0,
                    )
                    # Assignment is disabled in single-target 3-v-1, so slots
                    # 10 and 11 carry vertical belief information instead.
                    node[10] = relative_z / max(self.cfg.max_altitude, 1e-6)
                    node[11] = self.target_belief_velocity[i, target_id, 2] / max(
                        self.cfg.target_vertical_speed, 1e-6
                    )
            for i in range(self.cfg.n_uavs):
                for target_id in range(
                    min(graph["track_nodes"].shape[1], graph["target_nodes"].shape[1])
                ):
                    if graph["track_mask"][i, target_id]:
                        graph["target_nodes"][i, target_id] = graph["track_nodes"][i, target_id]
            obs["altitudes"] = self.altitudes.copy()
            obs["target_altitudes"] = self.target_altitudes.copy()
            obs["target_belief_altitude"] = self.target_belief_mean[:, :, 2].copy()
            obs["target_belief_vertical_velocity"] = self.target_belief_velocity[:, :, 2].copy()
        return obs

    def state_vector(self) -> np.ndarray:
        base = super().state_vector()
        if not hasattr(self, "altitudes"):
            return base
        capacity = max(int(self.cfg.building_state_capacity), int(self.cfg.building_count))
        building_heights = np.zeros(capacity, dtype=float)
        count = min(capacity, len(self.building_heights))
        building_heights[:count] = self.building_heights[:count]
        vertical = np.concatenate(
            [
                self.altitudes / max(self.cfg.max_altitude, 1e-6),
                self.vertical_velocities / max(self.cfg.vertical_speed, 1e-6),
                self.target_altitudes / max(self.cfg.max_altitude, 1e-6),
                self.target_vertical_velocity / max(self.cfg.target_vertical_speed, 1e-6),
                building_heights / max(self.cfg.max_altitude, 1e-6),
            ]
        )
        return np.concatenate([base, vertical])

    def state_dim(self) -> int:
        capacity = max(int(self.cfg.building_state_capacity), int(self.cfg.building_count))
        return (
            super().state_dim()
            + self.cfg.n_uavs * 2
            + self.cfg.n_targets * 2
            + capacity
        )


def _segment_rect_interval(start, end, rect):
    x0, y0, x1, y1 = rect
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    p = (-dx, dx, -dy, dy)
    q = (float(start[0] - x0), float(x1 - start[0]), float(start[1] - y0), float(y1 - start[1]))
    lo, hi = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
            continue
        ratio = qi / pi
        if pi < 0:
            lo = max(lo, ratio)
        else:
            hi = min(hi, ratio)
        if lo > hi:
            return None
    return lo, hi
