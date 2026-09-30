"""Post-detection single-evader pursuit mode.

The legacy cooperative-search environment remains a separate upstream module.
Each pursuit episode starts from a noisy target track handed off by that module,
after which one evader moves and three UAVs perform decentralized tracking and
capture.  Search-frontier features are therefore masked in this mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

try:
    from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
    from target_tracker import TrackMessage, TrackState, covariance_intersection, fuse_track_with_message
except ImportError:  # Allows ``import repro.pursuit_evasion_env``.
    from .cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
    from .target_tracker import TrackMessage, TrackState, covariance_intersection, fuse_track_with_message


class EvaderPolicy(Protocol):
    """Policy interface used by rule-based and learned evaders."""

    name: str

    def reset(self, env: "PursuitEvasionEnv") -> None: ...

    def action(self, env: "PursuitEvasionEnv", target_index: int, observation: np.ndarray) -> np.ndarray: ...


@dataclass
class PursuitConfig(WeakCommConfig):
    task_mode: str = "pursuit_core"
    n_uavs: int = 3
    n_targets: int = 1
    search_steps: int = 300
    # Cooperative pursuit assumes reliable sharing of teammate states.  Weak
    # communication remains available as an explicit ablation, but is not part
    # of the default pursuit difficulty.
    comm_mode: str = "full"
    outage_base_prob: float = 0.0
    outage_jammed_prob: float = 0.0
    # Random permanent failures are outside the current pursuit comparison
    # protocol. They must not confound policy quality with vehicle attrition.
    failure_base_prob: float = 0.0
    failure_jammed_prob: float = 0.0
    failure_recovery_prob: float = 0.0
    target_completion_steps: int = 10_000
    sensor_radius: float = 6.0
    entity_observation_radius: float = 7.0
    field_of_view_deg: float = 120.0
    # Pursuit starts after target acquisition.  A mission-level tracker keeps
    # the evader bearing/position observable to every active pursuer; buildings
    # remain physical constraints rather than turning pursuit back into search.
    pursuit_target_observable: bool = True
    # A search module has already found the target before the 3-v-1 episode.
    # The simulator exposes only a noisy handoff track, never target truth.
    handoff_initial_track: bool = True
    handoff_position_noise: float = 0.35
    handoff_position_variance: float = 0.25
    handoff_velocity_variance: float = 4.0
    pursuit_frontier_entities_enabled: bool = False
    building_count: int = 5
    # Fixed number of building slots exposed to the centralized critic so
    # randomized layouts retain a stable neural-network input dimension.
    building_state_capacity: int = 0
    building_min_size: float = 2.0
    building_max_size: float = 4.5
    # Default paper environment: the evader both repels nearby pursuers and
    # seeks the far side of buildings to create line-of-sight occlusion.
    evader_policy: str = "occlusion"  # random, repulsive, occlusion, external
    evader_turn_noise: float = 0.25
    evader_repulsion_radius: float = 10.0
    evader_occlusion_weight: float = 1.4
    capture_radius: float = 1.8
    capture_required_uavs: int = 2
    capture_hold_steps: int = 3
    capture_angular_span_deg: float = 100.0
    capture_reward: float = 100.0
    timeout_penalty: float = 50.0
    belief_process_noise: float = 0.12  # acceleration variance used by CV-KF
    belief_measurement_noise: float = 0.08
    belief_initial_velocity_variance: float = 4.0
    belief_max_age: int = 60
    belief_confidence_tau: float = 12.0
    track_lost_trace_threshold: float = 36.0
    information_gain_reward: float = 0.10
    information_gain_clip: float = 1.0
    uncertain_track_penalty: float = 0.05
    max_tracks_per_message: int = 2
    track_message_bytes: int = 192
    communication_cost_per_kb: float = 0.01
    # Journal-v1 distributed pursuit objective.  The legacy search reward is
    # still computed by the parent for diagnostics, but is not optimized in
    # pursuit mode.
    pursuit_reward_mode: str = "journal_v1"
    time_penalty: float = 0.05
    collision_event_penalty: float = 10.0
    safety_intervention_penalty: float = 0.10
    track_freshness_fusion_enabled: bool = True
    track_freshness_tau: float = 12.0
    track_confidence_floor: float = 0.05


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector / norm
    if fallback is not None:
        return _unit(np.asarray(fallback, dtype=float))
    return np.array([1.0, 0.0], dtype=float)


class RandomEvaderPolicy:
    name = "random"

    def __init__(self):
        self.headings: dict[int, np.ndarray] = {}

    def reset(self, env: "PursuitEvasionEnv") -> None:
        self.headings.clear()

    def action(self, env: "PursuitEvasionEnv", target_index: int, observation: np.ndarray) -> np.ndarray:
        previous = self.headings.get(target_index, env.target_velocity[target_index])
        noise = env.rng.normal(0.0, env.cfg.evader_turn_noise, size=2)
        heading = _unit(previous + noise)
        self.headings[target_index] = heading
        return heading


class RepulsiveEvaderPolicy:
    name = "repulsive"

    def reset(self, env: "PursuitEvasionEnv") -> None:
        return None

    def action(self, env: "PursuitEvasionEnv", target_index: int, observation: np.ndarray) -> np.ndarray:
        target = env.dynamic_targets[target_index]
        active = ~env.disabled_uavs
        delta = target[None, :] - env.positions[active].astype(float)
        distance = np.linalg.norm(delta, axis=1)
        relevant = distance < env.cfg.evader_repulsion_radius
        if np.any(relevant):
            weights = 1.0 / np.maximum(distance[relevant], 0.35) ** 2
            repulsion = np.sum(_unit_rows(delta[relevant]) * weights[:, None], axis=0)
        else:
            repulsion = env.target_velocity[target_index]
        return _unit(repulsion, env.target_velocity[target_index])


class OcclusionSeekingEvaderPolicy(RepulsiveEvaderPolicy):
    name = "occlusion"

    def action(self, env: "PursuitEvasionEnv", target_index: int, observation: np.ndarray) -> np.ndarray:
        repulsive = super().action(env, target_index, observation)
        target = env.dynamic_targets[target_index]
        active_positions = env.positions[~env.disabled_uavs].astype(float)
        if not env.buildings or not len(active_positions):
            return repulsive
        pursuer_center = np.mean(active_positions, axis=0)
        best_score = -float("inf")
        best_direction = repulsive
        for rect in env.buildings:
            x0, y0, x1, y1 = rect
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
            away_side = center + _unit(center - pursuer_center) * max(x1 - x0, y1 - y0)
            direction = _unit(away_side - target)
            candidate = target + direction * env.cfg.target_speed * 3.0
            blocked = sum(not env.has_line_of_sight(p, candidate) for p in active_positions)
            travel = float(np.linalg.norm(candidate - target))
            score = env.cfg.evader_occlusion_weight * blocked - 0.05 * travel
            if score > best_score:
                best_score = score
                best_direction = direction
        return _unit(0.65 * repulsive + 0.35 * best_direction)


class CallableEvaderPolicy:
    """Adapter for a learned policy callback returning a 2-D direction."""

    name = "external"

    def __init__(self, callback: Callable[[np.ndarray], np.ndarray]):
        self.callback = callback

    def reset(self, env: "PursuitEvasionEnv") -> None:
        return None

    def action(self, env: "PursuitEvasionEnv", target_index: int, observation: np.ndarray) -> np.ndarray:
        return _unit(np.asarray(self.callback(observation), dtype=float))


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


POLICY_FACTORIES = {
    "random": RandomEvaderPolicy,
    "repulsive": RepulsiveEvaderPolicy,
    "occlusion": OcclusionSeekingEvaderPolicy,
}


class PursuitEvasionEnv(WeakCommBeliefGraphEnv):
    """Weak-communication multi-UAV pursuit of one autonomous evader."""

    def action_dim(self) -> int:
        """Number of discrete motion actions for the legacy pursuit variant."""
        return 9

    def __init__(
        self,
        cfg: PursuitConfig | None = None,
        *,
        evader_policy: EvaderPolicy | Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        cfg = cfg or PursuitConfig()
        # Base construction calls observe_search() through reset(), so fields
        # used by overridden visibility methods must already exist.
        self.buildings: list[tuple[float, float, float, float]] = []
        super().__init__(cfg)
        self.cfg: PursuitConfig
        self.buildings = self._sample_buildings()
        self.evader_policy = self._resolve_evader_policy(evader_policy)
        self.evader_policy.reset(self)
        self._initialize_pursuit_state()

    def _initialize_pursuit_state(self) -> None:
        cfg = self.cfg
        self.capture_hold_count = 0
        self.captured = False
        self.ever_directly_seen = False
        self.was_directly_seen = False
        self.last_direct_detection_step = np.full((cfg.n_uavs, cfg.n_targets), -1, dtype=np.int32)
        # The pursuit block begins after the independent search block reports a
        # target.  A noisy track simulates that handoff without exposing truth.
        self.track_memory = [
            [TrackState.uninitialized(target_id) for target_id in range(cfg.n_targets)]
            for _ in range(cfg.n_uavs)
        ]
        self.target_belief_mean = np.zeros((cfg.n_uavs, cfg.n_targets, 2), dtype=float)
        self.target_belief_velocity = np.zeros((cfg.n_uavs, cfg.n_targets, 2), dtype=float)
        self.target_belief_covariance = np.repeat(
            np.eye(4, dtype=float)[None, None, :, :], cfg.n_uavs * cfg.n_targets, axis=0
        ).reshape(cfg.n_uavs, cfg.n_targets, 4, 4)
        self.target_belief_timestamp = np.full((cfg.n_uavs, cfg.n_targets), -1, dtype=np.int32)
        self.target_belief_source = np.full((cfg.n_uavs, cfg.n_targets), -1, dtype=np.int32)
        self.last_tracking_information_gain = 0.0
        self.last_track_messages = 0
        self.last_track_bytes = 0
        self.last_scheduled_track_ids: dict[int, list[int]] = {}
        # A single handed-off target requires no assignment decision: all three
        # pursuers are committed to target 0.  The assignment API remains for
        # legacy multi-target experiments.
        self.current_assignments = np.zeros(cfg.n_uavs, dtype=np.int32)
        self.last_assignment_switch_rate = 0.0
        self.capture_step = 0
        self._belief_updated_at = -1
        if cfg.handoff_initial_track:
            for target_id in range(cfg.n_targets):
                measurement = self.dynamic_targets[target_id] + self.rng.normal(
                    0.0, cfg.handoff_position_noise, size=2
                )
                measurement = np.clip(measurement, 0.0, cfg.grid_size - 1e-6)
                for uav_id in range(cfg.n_uavs):
                    self.track_memory[uav_id][target_id].initialize(
                        measurement,
                        timestamp=0,
                        measurement_variance=cfg.handoff_position_variance,
                        velocity_variance=cfg.handoff_velocity_variance,
                    )
                    # -2 denotes an external search-to-pursuit handoff.
                    self.target_belief_source[uav_id, target_id] = -2
            self.found_targets[:] = True
            self.target_first_discovery_step[:] = 0
            self.task_phase = "pursuit"
        self._update_target_beliefs(force=True)

    def reset(self) -> dict:
        """Reset base dynamics and every episode-local pursuit/track state."""
        obs = super().reset()
        # During base-class construction pursuit-specific fields do not exist.
        if hasattr(self, "evader_policy"):
            self.buildings = self._sample_buildings()
            self.evader_policy.reset(self)
            self._initialize_pursuit_state()
            obs = self.observe_search()
        return obs

    def _resolve_evader_policy(self, policy):
        if policy is None:
            try:
                return POLICY_FACTORIES[self.cfg.evader_policy]()
            except KeyError as exc:
                raise ValueError(f"Unknown evader_policy={self.cfg.evader_policy!r}") from exc
        if callable(policy) and not hasattr(policy, "action"):
            return CallableEvaderPolicy(policy)
        return policy

    def _sample_buildings(self) -> list[tuple[float, float, float, float]]:
        c = self.cfg
        rectangles = []
        protected = np.vstack([self.positions.astype(float), self.dynamic_targets])
        for _ in range(c.building_count * 12):
            if len(rectangles) >= c.building_count:
                break
            width, height = self.rng.uniform(c.building_min_size, c.building_max_size, size=2)
            x0 = float(self.rng.uniform(1.0, max(1.01, c.grid_size - width - 1.0)))
            y0 = float(self.rng.uniform(1.0, max(1.01, c.grid_size - height - 1.0)))
            rect = (x0, y0, x0 + width, y0 + height)
            if any(_point_in_rect(point, _expand_rect(rect, 0.8)) for point in protected):
                continue
            rectangles.append(rect)
        return rectangles

    def has_line_of_sight(self, start: np.ndarray, end: np.ndarray) -> bool:
        return not any(_segment_intersects_rect(start, end, rect) for rect in self.buildings)

    def direct_visibility_mask(self) -> np.ndarray:
        """Return [pursuer, target] visibility under range, FOV and LOS."""
        c = self.cfg
        if c.pursuit_target_observable:
            return np.broadcast_to(
                (~self.disabled_uavs)[:, None], (c.n_uavs, c.n_targets)
            ).copy()
        delta = self.dynamic_targets[None, :, :] - self.positions[:, None, :].astype(float)
        distance = np.linalg.norm(delta, axis=2)
        in_range = distance <= c.sensor_radius
        heading_vectors = _unit_rows(
            np.stack([self.uav_step_vector(int(h)) for h in self.headings]).astype(float)
        )
        direction = delta / np.maximum(distance[:, :, None], 1e-8)
        cosine = np.sum(direction * heading_vectors[:, None, :], axis=2)
        in_fov = cosine >= np.cos(np.deg2rad(c.field_of_view_deg * 0.5))
        los = np.ones_like(in_range)
        for i in range(c.n_uavs):
            for target_idx in range(c.n_targets):
                los[i, target_idx] = self.has_line_of_sight(
                    self.positions[i].astype(float), self.dynamic_targets[target_idx]
                )
        active = (~self.disabled_uavs)[:, None]
        return active & in_range & in_fov & los

    def causal_target_indices(self, i: int) -> np.ndarray:
        visible = self.direct_visibility_mask()
        causal = visible[i].copy()
        connected = np.flatnonzero(self.last_graph[i] > 0) if hasattr(self, "last_graph") else np.empty(0, dtype=int)
        if len(connected):
            causal |= np.any(visible[connected], axis=0)
        return np.flatnonzero(causal)

    def _scan_and_update(self, i: int) -> None:
        """Update maps and detections only inside UAV i's actual visual cone."""
        c = self.cfg
        origin = self.positions[i].astype(float)
        heading = _unit(self.uav_step_vector(int(self.headings[i])).astype(float))
        hit_delta = math.log(c.pf / c.pd)
        miss_delta = math.log((1.0 - c.pf) / (1.0 - c.pd))
        radius = int(math.ceil(c.sensor_radius))
        for y in range(max(0, int(origin[1]) - radius), min(c.grid_size, int(origin[1]) + radius + 1)):
            for x in range(max(0, int(origin[0]) - radius), min(c.grid_size, int(origin[0]) + radius + 1)):
                cell = np.array([x, y], dtype=float)
                delta = cell - origin
                distance = float(np.linalg.norm(delta))
                if distance > c.sensor_radius:
                    continue
                if distance > 1e-8:
                    cosine = float(np.dot(delta / distance, heading))
                    if cosine < np.cos(np.deg2rad(c.field_of_view_deg * 0.5)):
                        continue
                if not self.has_line_of_sight(origin, cell):
                    continue
                target_distance = np.linalg.norm(self.dynamic_targets - cell, axis=1)
                target_here = bool(np.any(target_distance <= 0.6))
                detected = self.rng.random() < (c.pd if target_here else c.pf)
                self.local_q[i, y, x] += hit_delta if detected else miss_delta
                self.local_belief_age_map[i, y, x] = 0.0
        visible = self.direct_visibility_mask()[i]
        if np.any(visible):
            self.found_targets |= visible

    def _update_tracking(self, prev_found: np.ndarray, prev_tracked: np.ndarray) -> dict:
        """Tracking metrics governed by direct FOV/LOS observations."""
        c = self.cfg
        visible_matrix = self.direct_visibility_mask()
        observed = np.any(visible_matrix, axis=0)
        distance = np.linalg.norm(
            self.positions[:, None, :].astype(float) - self.dynamic_targets[None, :, :],
            axis=2,
        )
        tracked_now = np.any(
            visible_matrix & (distance <= c.tracking_radius) & (~self.disabled_uavs[:, None]),
            axis=0,
        )
        self.found_targets |= observed
        newly_found = self.found_targets & ~prev_found
        self.target_first_discovery_step[newly_found] = self.t + 1
        previous_streak = self.target_tracking_streak.copy()
        self.tracked_targets = self.found_targets & tracked_now & ~self.completed_targets
        self.target_tracking_miss_count[self.tracked_targets] = 0
        self.target_tracking_streak[self.tracked_targets] += 1
        missed = (~self.completed_targets) & ~self.tracked_targets & (previous_streak > 0)
        self.target_tracking_miss_count[missed] += 1
        within_grace = missed & (self.target_tracking_miss_count <= c.tracking_grace_steps)
        self.target_tracking_streak[within_grace] = np.maximum(
            0,
            self.target_tracking_streak[within_grace] - max(int(c.tracking_streak_decay), 0),
        )
        confirmed_lost = missed & (self.target_tracking_miss_count > c.tracking_grace_steps)
        self.target_tracking_streak[confirmed_lost] = 0
        # Global target discovery belongs to the upstream search module.  Loss
        # of direct onboard visibility does not return this episode to search.
        self.task_phase = "pursuit"
        streak_delta = self.target_tracking_streak.astype(float) - previous_streak.astype(float)
        return {
            "new_discoveries": int(np.count_nonzero(newly_found)),
            "tracked_count": int(np.count_nonzero(self.tracked_targets)),
            "persistent_tracks": int(np.count_nonzero(self.tracked_targets & prev_tracked)),
            "streak_gain": float(np.sum(np.maximum(streak_delta, 0.0))),
            "streak_loss": float(np.sum(np.maximum(-streak_delta, 0.0))),
            "new_completions": 0,
            "lost_tracks": int(np.count_nonzero(confirmed_lost)),
            "escort_score": self._escort_score(distance),
            "discovery_rate": float(np.mean(self.found_targets)),
            "tracking_rate": float(np.mean(self.tracked_targets)),
            "completion_rate": float(self.captured) if hasattr(self, "captured") else 0.0,
        }

    def _update_target_beliefs(self, *, force: bool = False) -> None:
        if not force and self._belief_updated_at == self.t:
            return
        c = self.cfg
        dt = float(getattr(self, "current_decision_dt", c.decision_dt))
        for tracks in self.track_memory:
            for track in tracks:
                track.predict(dt, c.belief_process_noise)

        visible = self.direct_visibility_mask()
        information_gain = 0.0
        # Sensor noise is sampled once by each physical observer.  Receivers do
        # not manufacture new measurements; they receive a timestamped track.
        for i in range(c.n_uavs):
            for target_id in np.flatnonzero(visible[i]):
                measurement = self.dynamic_targets[target_id] + self.rng.normal(
                    0.0, c.belief_measurement_noise, size=2
                )
                information_gain += self.track_memory[i][target_id].update_xy(
                    measurement,
                    self.t,
                    c.belief_measurement_noise**2,
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
            # One top-K schedule is chosen per sender, then broadcast to its
            # connected neighbors. Priority combines receiver AoI with the
            # covariance reduction that CI would provide.
            for source in range(c.n_uavs):
                if self.disabled_uavs[source]:
                    continue
                receivers = [
                    int(receiver) for receiver in np.flatnonzero(self.last_graph[:, source] > 0)
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
                            receiver_message.mean, receiver_message.covariance,
                            source_message.mean, source_message.covariance,
                            grid_size=21,
                        )
                        reduction = max(
                            0.0,
                            float(np.trace(receiver_message.covariance[:2, :2]))
                            - float(np.trace(fused_covariance[:2, :2])),
                        )
                        score = max(
                            score,
                            (1.0 + min(max(self.t - receiver_message.timestamp, 0), c.belief_max_age))
                            * reduction,
                        )
                    priorities.append((score, target_id))
                priorities.sort(key=lambda item: (-item[0], item[1]))
                limit = max(0, min(int(c.max_tracks_per_message), len(priorities)))
                selected = [target_id for _, target_id in priorities[:limit]]
                scheduled[source] = selected
                for receiver in receivers:
                    for target_id in selected:
                        message = messages[(source, target_id)]
                        information_gain += fuse_track_with_message(
                            self.track_memory[receiver][target_id],
                            message,
                            now=self.t if c.track_freshness_fusion_enabled else None,
                            freshness_tau=(
                                c.track_freshness_tau
                                if c.track_freshness_fusion_enabled
                                else None
                            ),
                            link_confidence=float(
                                np.clip(self.last_graph[receiver, source], 0.0, 1.0)
                            ),
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
        for i, tracks in enumerate(self.track_memory):
            for target_id, track in enumerate(tracks):
                if not track.initialized:
                    self.target_belief_timestamp[i, target_id] = -1
                    continue
                self.target_belief_mean[i, target_id] = track.mean[:2]
                self.target_belief_velocity[i, target_id] = track.mean[2:]
                self.target_belief_covariance[i, target_id] = track.covariance
                self.target_belief_timestamp[i, target_id] = track.timestamp

    def target_belief_age(self, i: int, target_id: int = 0) -> float:
        return self.track_memory[i][target_id].age(self.t, self.cfg.belief_max_age)

    def set_assignments(self, assignments: np.ndarray) -> None:
        """Set target IDs selected by the policy; ``n_targets`` means search."""
        assignments = np.asarray(assignments, dtype=np.int32)
        if assignments.shape != (self.cfg.n_uavs,):
            raise ValueError(f"Expected assignments {(self.cfg.n_uavs,)}, got {assignments.shape}")
        if np.any((assignments < 0) | (assignments > self.cfg.n_targets)):
            raise ValueError("Assignment must be a target ID or the search slot")
        self.last_assignment_switch_rate = float(
            np.mean(assignments != self.current_assignments)
        )
        self.current_assignments = assignments.copy()

    def _track_evaluation_metrics(self) -> dict[str, float]:
        """Ground-truth diagnostics used for evaluation, never actor input."""
        position_errors = []
        velocity_errors = []
        nees_values = []
        ages = []
        for tracks in self.track_memory:
            for target_id, track in enumerate(tracks):
                if not track.initialized:
                    continue
                position_error = track.mean[:2] - self.dynamic_targets[target_id]
                velocity_error = track.mean[2:] - self.target_velocity[target_id]
                position_errors.append(float(np.linalg.norm(position_error)))
                velocity_errors.append(float(np.linalg.norm(velocity_error)))
                ages.append(track.age(self.t, self.cfg.belief_max_age))
                try:
                    nees_values.append(
                        float(position_error @ np.linalg.solve(track.covariance[:2, :2], position_error))
                    )
                except np.linalg.LinAlgError:
                    nees_values.append(float("nan"))
        finite_nees = np.asarray(nees_values, dtype=float)
        finite_nees = finite_nees[np.isfinite(finite_nees)]
        return {
            "track_position_rmse": float(
                np.sqrt(np.mean(np.square(position_errors))) if position_errors else np.nan
            ),
            "track_velocity_rmse": float(
                np.sqrt(np.mean(np.square(velocity_errors))) if velocity_errors else np.nan
            ),
            "mean_track_nees": float(np.mean(finite_nees) if len(finite_nees) else np.nan),
            # 95% chi-square consistency interval for a 2-D position error.
            "track_nees_consistency_rate": float(
                np.mean((finite_nees >= 0.0506) & (finite_nees <= 7.3778))
                if len(finite_nees)
                else np.nan
            ),
            "mean_track_age": float(np.mean(ages) if ages else self.cfg.belief_max_age),
            "p95_track_age": float(np.percentile(ages, 95) if ages else self.cfg.belief_max_age),
        }

    def evader_observation(self, target_index: int = 0) -> np.ndarray:
        """Egocentric fixed-size observation for rule and PPO evaders."""
        c = self.cfg
        target = self.dynamic_targets[target_index]
        relative = self.positions.astype(float) - target[None, :]
        distance = np.linalg.norm(relative, axis=1)
        order = np.argsort(distance)
        pursuers = []
        for idx in order[: c.n_uavs]:
            visible = distance[idx] <= c.sensor_radius and self.has_line_of_sight(target, self.positions[idx])
            pursuers.extend(
                [
                    relative[idx, 0] / c.grid_size if visible else 0.0,
                    relative[idx, 1] / c.grid_size if visible else 0.0,
                    min(distance[idx] / c.grid_size, 1.0) if visible else 1.0,
                    float(visible),
                ]
            )
        boundary = np.array(
            [target[0], target[1], c.grid_size - target[0], c.grid_size - target[1]]
        ) / c.grid_size
        nearest_building = np.zeros(4, dtype=float)
        if self.buildings:
            centers = np.array([[(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5] for r in self.buildings])
            rel = centers - target
            idx = int(np.argmin(np.linalg.norm(rel, axis=1)))
            nearest_building[:2] = rel[idx] / c.grid_size
            nearest_building[2:] = [
                (self.buildings[idx][2] - self.buildings[idx][0]) / c.grid_size,
                (self.buildings[idx][3] - self.buildings[idx][1]) / c.grid_size,
            ]
        return np.asarray(
            [
                target[0] / c.grid_size,
                target[1] / c.grid_size,
                self.target_velocity[target_index, 0] / max(c.target_speed, 1e-6),
                self.target_velocity[target_index, 1] / max(c.target_speed, 1e-6),
                *pursuers,
                *boundary,
                *nearest_building,
            ],
            dtype=np.float32,
        )

    def _move_dynamic_entities(self) -> None:
        if self.cfg.dynamic_targets_enabled and not self.captured:
            for target_idx in range(self.cfg.n_targets):
                direction = self.evader_policy.action(
                    self, target_idx, self.evader_observation(target_idx)
                )
                self.target_velocity[target_idx] = _unit(direction) * self.cfg.target_speed
            self.dynamic_targets, self.target_velocity = self._move_points_avoiding_buildings(
                self.dynamic_targets, self.target_velocity
            )
        if self.cfg.dynamic_obstacles_enabled:
            self.obstacles, self.obstacle_velocity = self._move_points(
                self.obstacles, self.obstacle_velocity
            )
        self.target_grid = self._target_occupancy()

    def _move_points_avoiding_buildings(self, points, velocity):
        dt = float(getattr(self, "current_decision_dt", self.cfg.decision_dt))
        proposed = points + velocity * dt
        for idx in range(len(points)):
            if any(_point_in_rect(proposed[idx], _expand_rect(rect, 0.2)) for rect in self.buildings):
                velocity[idx] *= -1.0
                proposed[idx] = points[idx] + velocity[idx] * dt
        return self._move_points(points, velocity)

    def _capture_geometry(self) -> tuple[bool, int, float]:
        target = self.dynamic_targets[0]
        delta = self.positions.astype(float) - target[None, :]
        distance = np.linalg.norm(delta, axis=1)
        close = np.flatnonzero((distance <= self.cfg.capture_radius) & ~self.disabled_uavs)
        if len(close) < self.cfg.capture_required_uavs:
            return False, len(close), 0.0
        angles = np.sort(np.mod(np.arctan2(delta[close, 1], delta[close, 0]), 2 * np.pi))
        wrapped = np.r_[angles, angles[0] + 2 * np.pi]
        largest_gap = float(np.max(np.diff(wrapped)))
        span = 2 * np.pi - largest_gap
        required_span = np.deg2rad(self.cfg.capture_angular_span_deg)
        return span >= required_span, len(close), float(np.rad2deg(span))

    def step_joint(self, search_actions: np.ndarray) -> dict:
        was_visible = bool(np.any(self.direct_visibility_mask()))
        result = super().step_joint(search_actions)
        # ``step_search`` constructs the post-transition observation, which
        # already advances and updates tracks for this timestamp.
        self._update_target_beliefs()
        now_visible = bool(np.any(self.direct_visibility_mask()))
        newly_detected = now_visible and not self.ever_directly_seen
        reacquired = now_visible and self.ever_directly_seen and not was_visible
        lost = was_visible and not now_visible
        self.ever_directly_seen |= now_visible
        geometry_ok, close_count, angular_span = self._capture_geometry()
        self.capture_hold_count = self.capture_hold_count + 1 if geometry_ok else 0
        just_captured = self.capture_hold_count >= self.cfg.capture_hold_steps and not self.captured
        self.captured |= just_captured
        if just_captured:
            self.capture_step = int(self.t)
        initialized_tracks = [
            track for tracks in self.track_memory for track in tracks if track.initialized
        ]
        valid_tracks = sum(
            track.position_trace() <= self.cfg.track_lost_trace_threshold
            and track.age(self.t, self.cfg.belief_max_age) < self.cfg.belief_max_age
            for track in initialized_tracks
        )
        mean_trace = float(
            np.mean([track.position_trace() for track in initialized_tracks])
            if initialized_tracks else self.cfg.track_lost_trace_threshold
        )
        uncertainty_cost = min(
            mean_trace / max(self.cfg.track_lost_trace_threshold, 1e-6), 2.0
        )
        reward_components = {
            # Pursuit begins after a noisy target-track handoff. Track
            # availability is part of the task definition, not an event that
            # the policy should receive reward for creating. Detection,
            # visibility and reacquisition remain diagnostics only.
            "capture": self.cfg.capture_reward * float(just_captured),
            "information_gain": self.cfg.information_gain_reward
            * min(self.last_tracking_information_gain, self.cfg.information_gain_clip),
            "uncertainty": -self.cfg.uncertain_track_penalty * uncertainty_cost,
            "communication": -self.cfg.communication_cost_per_kb
            * self.last_track_bytes
            / 1024.0,
            "collision": -self.cfg.collision_event_penalty
            * float(result.get("collisions", 0)),
            "safety_intervention": -self.cfg.safety_intervention_penalty
            * float(result.get("safety_interventions", 0)),
            "time": -self.cfg.time_penalty,
            "timeout": -self.cfg.timeout_penalty
            * float(self.t >= self.cfg.search_steps and not self.captured),
        }
        if self.cfg.pursuit_reward_mode != "journal_v1":
            raise ValueError(
                f"Unknown pursuit_reward_mode={self.cfg.pursuit_reward_mode!r}; "
                "expected 'journal_v1'."
            )
        pursuit_bonus = reward_components["capture"] + reward_components["timeout"]
        estimation_bonus = sum(
            reward_components[key]
            for key in ("information_gain", "uncertainty", "communication")
        )
        result["reward"] = float(sum(reward_components.values()))
        result["capture_success"] = float(self.captured)
        result["capture_hold_count"] = int(self.capture_hold_count)
        result["capture_close_uavs"] = int(close_count)
        result["capture_angular_span_deg"] = angular_span
        result["direct_target_visible"] = float(now_visible)
        result["target_reacquired"] = float(reacquired)
        result["target_lost"] = float(lost)
        result["pursuit_bonus"] = float(pursuit_bonus)
        result["tracking_information_gain"] = float(self.last_tracking_information_gain)
        result["mean_track_position_trace"] = mean_trace
        result["valid_track_rate"] = float(valid_tracks / max(self.cfg.n_uavs * self.cfg.n_targets, 1))
        result["track_messages"] = int(self.last_track_messages)
        result["track_communication_bytes"] = int(self.last_track_bytes)
        result["scheduled_track_ids"] = {
            int(source): list(target_ids) for source, target_ids in self.last_scheduled_track_ids.items()
        }
        result["assignments"] = self.current_assignments.astype(int).tolist()
        result["assignment_search_rate"] = float(
            np.mean(self.current_assignments == self.cfg.n_targets)
        )
        result["assignment_switch_rate"] = self.last_assignment_switch_rate
        for target_id in range(self.cfg.n_targets):
            result[f"assignment_target_{target_id}_rate"] = float(
                np.mean(self.current_assignments == target_id)
            )
        result["estimation_bonus"] = float(estimation_bonus)
        result["reward_components"] = {
            **{key: float(value) for key, value in reward_components.items()},
            "pursuit": float(pursuit_bonus),
            "estimation": float(estimation_bonus),
        }
        result.update(self._track_evaluation_metrics())
        result["capture_time"] = int(self.capture_step)
        result["terminated"] = bool(self.captured)
        result["truncated"] = bool(self.t >= self.cfg.search_steps and not self.captured)
        result["target_completion_rate"] = float(self.captured)
        result["obs"] = self.observe_search()
        return result

    def observe_search(self) -> dict:
        if hasattr(self, "target_belief_mean"):
            self._update_target_beliefs()
        obs = super().observe_search()
        if hasattr(self, "target_belief_mean"):
            graph = obs["hetero_graph"]
            if not self.cfg.pursuit_frontier_entities_enabled:
                graph["frontier_nodes"].fill(0.0)
                graph["frontier_mask"].fill(False)
                # These flat-observation slots are legacy TPM frontier/revisit
                # descriptors.  Neutralize them for the post-detection task.
                obs["agent_observations"][:, 23:32] = 0.0
            direct_visibility = self.direct_visibility_mask()
            graph["track_nodes"].fill(0.0)
            graph["track_mask"].fill(False)
            for i in range(self.cfg.n_uavs):
                for target_id, track in enumerate(self.track_memory[i]):
                    if target_id >= graph["track_nodes"].shape[1] or not track.initialized:
                        continue
                    age = self.target_belief_age(i, target_id)
                    confidence = float(np.exp(-age / max(self.cfg.belief_confidence_tau, 1e-6)))
                    # The 3-D extension stores xyz beliefs; the shared planar
                    # graph encoder consumes the horizontal projection and is
                    # augmented with vertical channels by its subclass.
                    delta = self.target_belief_mean[i, target_id, :2] - self.positions[i].astype(float)
                    node = graph["track_nodes"][i, target_id]
                    node[:5] = [
                        delta[0] / self.cfg.grid_size,
                        delta[1] / self.cfg.grid_size,
                        self.target_belief_velocity[i, target_id, 0] / max(self.cfg.uav_speed, 1e-6),
                        self.target_belief_velocity[i, target_id, 1] / max(self.cfg.uav_speed, 1e-6),
                        min(float(np.linalg.norm(delta)) / self.cfg.grid_size, 1.0),
                    ]
                    position_logdet = track.position_logdet()
                    position_trace = track.position_trace()
                    node[5] = confidence * track.existence_probability
                    node[6] = float(np.clip(position_logdet / 8.0, -1.0, 1.0))
                    node[7] = float(np.clip(position_trace / max(self.cfg.track_lost_trace_threshold, 1e-6), 0.0, 2.0))
                    node[8] = float(direct_visibility[i, target_id])
                    node[9] = float(position_trace <= self.cfg.track_lost_trace_threshold)
                    node[10] = float(
                        np.count_nonzero(self.current_assignments == target_id)
                        / max(self.cfg.n_uavs, 1)
                    )
                    node[11] = float(position_trace > self.cfg.track_lost_trace_threshold)
                    node[12] = min(age / max(self.cfg.belief_max_age, 1), 1.0)
                    graph["track_mask"][i, target_id] = True
            # Refresh the deprecated combined slots after replacing physical
            # observations with persistent local track estimates.
            graph["target_nodes"].fill(0.0)
            graph["target_mask"].fill(False)
            for i in range(self.cfg.n_uavs):
                slot = 0
                for target_id in np.flatnonzero(graph["track_mask"][i]):
                    if slot >= graph["target_nodes"].shape[1]:
                        break
                    graph["target_nodes"][i, slot] = graph["track_nodes"][i, target_id]
                    graph["target_mask"][i, slot] = True
                    slot += 1
            obs["target_belief_mean"] = self.target_belief_mean.copy()
            obs["target_belief_velocity"] = self.target_belief_velocity.copy()
            obs["target_belief_covariance"] = self.target_belief_covariance.copy()
            obs["target_belief_timestamp"] = self.target_belief_timestamp.copy()
            obs["target_track_initialized"] = np.asarray(
                [[track.initialized for track in tracks] for tracks in self.track_memory], dtype=bool
            )
            obs["direct_visibility"] = direct_visibility
        return obs

    def state_vector(self) -> np.ndarray:
        """Centralized-training state with fixed-size building geometry."""
        if not hasattr(self, "obstacles"):
            return np.zeros(self.state_dim(), dtype=np.float32)
        base = super().state_vector()
        capacity = max(int(self.cfg.building_state_capacity), int(self.cfg.building_count))
        geometry = np.zeros((capacity, 4), dtype=np.float32)
        for idx, rect in enumerate(self.buildings[:capacity]):
            geometry[idx] = np.asarray(rect, dtype=np.float32) / self.cfg.grid_size
        return np.concatenate([base, geometry.ravel()])

    def state_dim(self) -> int:
        capacity = max(int(self.cfg.building_state_capacity), int(self.cfg.building_count))
        return super().state_dim() + capacity * 4


def _expand_rect(rect, margin):
    x0, y0, x1, y1 = rect
    return x0 - margin, y0 - margin, x1 + margin, y1 + margin


def _point_in_rect(point, rect) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= float(point[0]) <= x1 and y0 <= float(point[1]) <= y1


def _segment_intersects_rect(start, end, rect) -> bool:
    """Liang-Barsky segment/AABB intersection."""
    x0, y0, x1, y1 = rect
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    p = (-dx, dx, -dy, dy)
    q = (float(start[0] - x0), float(x1 - start[0]), float(start[1] - y0), float(y1 - start[1]))
    lo, hi = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            lo = max(lo, ratio)
        else:
            hi = min(hi, ratio)
        if lo > hi:
            return False
    return True
