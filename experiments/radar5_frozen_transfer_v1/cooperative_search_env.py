"""Interference-outage belief-graph environment.

This environment combines three ideas:

- Wang et al. 2025: TPM search under sweeping interference.
- Zhao et al. 2025: local dynamic graph attention over variable neighbors.
- Chang et al. 2025: DWA-style local obstacle avoidance with inter-UAV safety.

It is an integrated research environment, not a strict reproduction of any
single paper's full experimental protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from tpm_search_base_env import DIRECTIONS, WangConfig, WangSearchInterferenceEnv, log_odds_to_prob
except ImportError:  # Allows `import repro.cooperative_search_env` from the workspace root.
    from .tpm_search_base_env import DIRECTIONS, WangConfig, WangSearchInterferenceEnv, log_odds_to_prob


@dataclass
class WeakCommConfig(WangConfig):
    n_uavs: int = 6
    n_targets: int = 15
    grid_size: int = 25
    search_steps: int = 200
    comm_mode: str = "outage"  # full, outage, none
    avoidance_mode: str = "dwa_orca"  # none, dwa, dwa_orca
    dynamic_targets_enabled: bool = True
    dynamic_obstacles_enabled: bool = True
    allow_hover: bool = True
    initial_formation: str = "cluster"  # cluster, corners
    formation_spacing: float = 2.0
    n_obstacles: int = 10
    obstacle_radius: float = 1.25
    uav_collision_radius: float = 0.8
    safe_radius: float = 1.6
    sensor_radius: float = 2.0
    tracking_radius: float = 1.75
    target_speed: float = 0.75
    obstacle_speed: float = 0.5
    comm_radius: float = 10.0
    graph_beta: float = 0.28
    outage_base_prob: float = 0.02
    outage_jammed_prob: float = 0.45
    outage_recovery_prob: float = 0.45
    failure_base_prob: float = 0.0008
    failure_jammed_prob: float = 0.008
    failure_recovery_prob: float = 0.02
    jammer_effect_radius: float = 8.0
    disabled_action_penalty: float = 0.5
    age_decay: float = 0.82
    invalid_action_penalty: float = 1.0
    obstacle_penalty: float = 8.0
    pair_penalty: float = 4.0
    tracking_reward: float = 0.5
    # target_completion_steps <= 0 selects discovery-only training.  In that
    # mode the three rewards below make first detections dominate generic map
    # exploration, while the convex milestone term makes the last few targets
    # worth more than the easy early ones.
    discovery_reward: float = 20.0
    discovery_progress_reward: float = 1.5
    discovery_milestone_reward: float = 10.0
    all_targets_discovered_reward: float = 50.0
    search_step_penalty: float = 0.01
    reward_mode: str = "team_potential"  # team_potential, legacy_shaped
    discovery_potential_weight: float = 1.0
    discovery_tail_exponent: float = 2.0
    exploration_potential_weight: float = 0.30
    safety_cost_weight: float = 0.10
    tracking_stage_threshold: float = 0.35
    persistent_tracking_reward: float = 1.5
    tracking_streak_reward: float = 1.5
    tracking_streak_loss_penalty: float = 2.0
    completion_reward: float = 30.0
    target_lost_penalty: float = 10.0
    escort_reward: float = 0.2
    target_progress_reward: float = 0.5
    coverage_gain_reward: float = 20.0
    new_cell_reward: float = 0.05
    revisit_gain_reward: float = 0.05
    revisit_map_enabled: bool = True
    freshness_fusion_enabled: bool = True
    belief_fusion_mode: str = "soft"  # soft, hard
    belief_fusion_temperature: float = 0.35
    belief_freshness_tau: float = 8.0
    belief_confidence_floor: float = 0.05
    max_belief_age: float = 100.0
    # Communication-history ablation:
    # none     -> physical-link mask only, equal source weights;
    # edge_age -> distance/AoI-weighted source selection, no AoI penalty on revisit memory;
    # full     -> edge_age plus freshness-aware revisit-memory fusion.
    history_mode: str = "full"
    revisit_aging_rate: float = 0.035
    revisit_diffusion: float = 0.08
    revisit_fresh_threshold: float = 0.35
    revisit_reward_threshold: float = 0.65
    stagnation_penalty: float = 0.05
    long_stagnation_penalty: float = 0.15
    stagnation_threshold: int = 5
    long_stagnation_threshold: int = 10
    crash_penalty: float = 30.0
    uav_collision_penalty: float = 20.0
    hetero_target_nodes: int = 6
    # Tracking and search-frontier entities have different semantics and are
    # encoded separately. ``hetero_target_nodes`` remains the frontier default
    # for backward-compatible experiment configurations.
    hetero_track_nodes: int = 0  # <= 0 uses n_targets
    hetero_frontier_nodes: int = 6
    hetero_obstacle_nodes: int = 4
    entity_observation_radius: float = 6.0
    target_completion_steps: int = 0
    tracking_grace_steps: int = 3
    tracking_streak_decay: int = 1
    target_prediction_weight: float = 0.75
    dynamic_dt_enabled: bool = True
    min_decision_dt: float = 0.45
    max_decision_dt: float = 1.0
    obstacle_dt_radius_factor: float = 3.0
    pair_dt_radius_factor: float = 2.5
    dynamic_dt_smoothing: float = 0.55
    avoidance_prediction_horizon: int = 3
    obstacle_prediction_buffer: float = 0.25
    pair_prediction_buffer: float = 0.15


class WeakCommBeliefGraphEnv(WangSearchInterferenceEnv):
    """Wang search environment with local belief graph and DWA-ORCA safety."""

    cfg: WeakCommConfig

    def __init__(self, cfg: WeakCommConfig | None = None):
        super().__init__(cfg or WeakCommConfig())

    def reset(self) -> dict:
        obs = super().reset()
        c = self.cfg
        self.dynamic_targets = self.targets.astype(float) + self.rng.uniform(-0.15, 0.15, size=(c.n_targets, 2))
        self.target_velocity = self._random_velocities(c.n_targets, c.target_speed)
        if not c.dynamic_targets_enabled:
            self.target_velocity[:] = 0.0
        self.found_targets = np.zeros(c.n_targets, dtype=bool)
        # -1 means right-censored (not discovered within the episode).  The
        # value is the one-based decision step of the first team detection.
        self.target_first_discovery_step = np.full(c.n_targets, -1, dtype=np.int32)
        self.tracked_targets = np.zeros(c.n_targets, dtype=bool)
        self.prev_tracked_targets = np.zeros(c.n_targets, dtype=bool)
        self.target_tracking_streak = np.zeros(c.n_targets, dtype=int)
        self.target_tracking_miss_count = np.zeros(c.n_targets, dtype=int)
        self.completed_targets = np.zeros(c.n_targets, dtype=bool)
        self.task_phase = "search"
        if c.initial_formation == "cluster":
            self.positions = self._cluster_initial_positions()
            self.headings = self._initial_headings_from_prior()
        self.target_grid = self._target_occupancy()
        self.obstacles = self._sample_obstacles(c.n_obstacles)
        self.obstacle_velocity = self._random_velocities(len(self.obstacles), c.obstacle_speed)
        if not c.dynamic_obstacles_enabled:
            self.obstacle_velocity[:] = 0.0
        self.belief_age = np.zeros(c.n_uavs, dtype=float)
        # Cell-wise age of information (AoI). A scalar age cannot distinguish
        # a freshly sensed local patch from stale cells elsewhere in the map.
        self.local_belief_age_map = np.zeros(
            (c.n_uavs, c.grid_size, c.grid_size), dtype=np.float32
        )
        self.last_graph = np.zeros((c.n_uavs, c.n_uavs), dtype=float)
        self.disconnected_uavs = np.zeros(c.n_uavs, dtype=bool)
        self.disabled_uavs = np.zeros(c.n_uavs, dtype=bool)
        self.crashed_uavs = np.zeros(c.n_uavs, dtype=bool)
        self.interference_level = np.zeros(c.n_uavs, dtype=float)
        self.current_decision_dt = c.decision_dt
        self.dynamic_dt_complexity = 0.0
        self.visit_counts = np.zeros((c.grid_size, c.grid_size), dtype=np.int32)
        # Third local search-map layer. 0 means recently observed and 1 means
        # stale/unvisited. Each UAV owns its map; peer observations enter only
        # through the instantaneous communication graph.
        initial_revisit = 1.0 if c.revisit_map_enabled else 0.0
        self.local_revisit_map = np.full(
            (c.n_uavs, c.grid_size, c.grid_size), initial_revisit, dtype=np.float32
        )
        self.stagnation_steps = np.zeros(c.n_uavs, dtype=np.int32)
        self.agent_distance_travelled = np.zeros(c.n_uavs, dtype=float)
        for pos in self.positions:
            x, y = np.clip(pos.astype(int), 0, c.grid_size - 1)
            self.visit_counts[y, x] += 1
        for i in range(c.n_uavs):
            self._mark_revisit_footprint_fresh(i)
        self._update_outage_state()
        if c.comm_mode == "full":
            active = (~self.disabled_uavs).astype(float)
            self.last_graph = (
                active[:, None] * active[None, :]
            ) * (1.0 - np.eye(c.n_uavs))
        elif c.comm_mode == "none":
            self.last_graph.fill(0.0)
        elif c.comm_mode == "outage":
            self.last_graph = self._comm_graph()
        else:
            raise ValueError(
                f"Unknown comm_mode={c.comm_mode!r}; expected 'full', 'outage', or 'none'."
            )
        self._update_dynamic_dt()
        return self.observe_search()

    def _cluster_initial_positions(self) -> np.ndarray:
        c = self.cfg
        cols = int(math.ceil(math.sqrt(c.n_uavs)))
        rows = int(math.ceil(c.n_uavs / cols))
        spacing = max(1.0, float(c.formation_spacing))
        width = (cols - 1) * spacing
        height = (rows - 1) * spacing
        origin = np.array(
            [
                max(1.0, min(c.grid_size - 2.0 - width, c.grid_size * 0.18)),
                max(1.0, min(c.grid_size - 2.0 - height, c.grid_size * 0.18)),
            ],
            dtype=float,
        )
        positions = []
        for idx in range(c.n_uavs):
            row, col = divmod(idx, cols)
            positions.append(origin + np.array([col * spacing, row * spacing], dtype=float))
        return np.asarray(positions, dtype=float)

    def _sample_obstacles(self, count: int) -> np.ndarray:
        c = self.cfg
        blocked = {tuple(p) for p in self.positions.tolist()}
        blocked.update(tuple(p) for p in self.targets.tolist())
        obstacles = []
        attempts = 0
        while len(obstacles) < count and attempts < count * 60:
            attempts += 1
            cell = tuple(self.rng.integers(1, c.grid_size - 1, size=2).tolist())
            if cell in blocked:
                continue
            if np.min(np.linalg.norm(self.positions.astype(float) - np.asarray(cell, dtype=float), axis=1)) < c.obstacle_radius + c.safe_radius:
                continue
            if any(np.linalg.norm(np.asarray(cell) - np.asarray(o)) < c.obstacle_radius * 2 for o in obstacles):
                continue
            obstacles.append(cell)
            blocked.add(cell)
        return np.asarray(obstacles, dtype=float)

    def _random_velocities(self, count: int, speed: float) -> np.ndarray:
        if count == 0:
            return np.empty((0, 2), dtype=float)
        angles = self.rng.uniform(0.0, 2.0 * math.pi, size=count)
        speeds = self.rng.uniform(0.35 * speed, speed, size=count)
        return np.column_stack([np.cos(angles), np.sin(angles)]) * speeds[:, None]

    def observe_search(self) -> dict:
        base = super().observe_search()
        base.update(
            {
                "obstacles": self.obstacles.copy() if hasattr(self, "obstacles") else np.empty((0, 2)),
                "obstacle_velocity": self.obstacle_velocity.copy()
                if hasattr(self, "obstacle_velocity")
                else np.empty((0, 2)),
                "dynamic_targets": self.dynamic_targets.copy()
                if hasattr(self, "dynamic_targets")
                else np.empty((0, 2)),
                "target_velocity": self.target_velocity.copy() if hasattr(self, "target_velocity") else np.empty((0, 2)),
                "found_targets": self.found_targets.copy() if hasattr(self, "found_targets") else np.zeros(self.cfg.n_targets),
                "tracked_targets": self.tracked_targets.copy()
                if hasattr(self, "tracked_targets")
                else np.zeros(self.cfg.n_targets),
                "completed_targets": self.completed_targets.copy()
                if hasattr(self, "completed_targets")
                else np.zeros(self.cfg.n_targets),
                "target_tracking_streak": self.target_tracking_streak.copy()
                if hasattr(self, "target_tracking_streak")
                else np.zeros(self.cfg.n_targets),
                "belief_age": self.belief_age.copy() if hasattr(self, "belief_age") else np.zeros(self.cfg.n_uavs),
                "disconnected_uavs": self.disconnected_uavs.copy()
                if hasattr(self, "disconnected_uavs")
                else np.zeros(self.cfg.n_uavs, dtype=bool),
                "disabled_uavs": self.disabled_uavs.copy()
                if hasattr(self, "disabled_uavs")
                else np.zeros(self.cfg.n_uavs, dtype=bool),
                "crashed_uavs": self.crashed_uavs.copy()
                if hasattr(self, "crashed_uavs")
                else np.zeros(self.cfg.n_uavs, dtype=bool),
                "interference_level": self.interference_level.copy()
                if hasattr(self, "interference_level")
                else np.zeros(self.cfg.n_uavs),
                "local_revisit_map": self.local_revisit_map.copy()
                if hasattr(self, "local_revisit_map")
                else np.ones((self.cfg.n_uavs, self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32),
                "local_belief_age_map": self.local_belief_age_map.copy()
                if hasattr(self, "local_belief_age_map")
                else np.zeros(
                    (self.cfg.n_uavs, self.cfg.grid_size, self.cfg.grid_size),
                    dtype=np.float32,
                ),
                "graph_features": self.graph_features() if hasattr(self, "obstacles") else np.zeros((self.cfg.n_uavs, 32)),
                "state_vector": self.state_vector() if hasattr(self, "obstacles") else np.zeros(self.state_dim()),
                "agent_observations": self.agent_observation_vectors()
                if hasattr(self, "obstacles")
                else np.zeros((self.cfg.n_uavs, self.obs_dim())),
                "hetero_graph": self.heterogeneous_graph_observation()
                if hasattr(self, "obstacles")
                else self._empty_heterogeneous_graph(),
                "comm_adjacency": self.comm_adjacency(),
                "current_decision_dt": float(getattr(self, "current_decision_dt", self.cfg.decision_dt)),
                "dynamic_dt_complexity": float(getattr(self, "dynamic_dt_complexity", 0.0)),
            }
        )
        return base

    def _update_dynamic_dt(self) -> None:
        c = self.cfg
        if not c.dynamic_dt_enabled:
            self.current_decision_dt = c.decision_dt
            self.dynamic_dt_complexity = 0.0
            return
        max_dt = min(c.max_decision_dt, c.decision_dt)
        min_dt = min(c.min_decision_dt, max_dt)
        obstacle_pressure = self._obstacle_pressure()
        pair_pressure = self._pair_pressure()
        jam_pressure = float(np.mean(getattr(self, "interference_level", np.zeros(c.n_uavs))))
        tracking_pressure = 0.35 if getattr(self, "task_phase", "search") == "tracking" else 0.0
        complexity = float(np.clip(0.42 * obstacle_pressure + 0.28 * pair_pressure + 0.18 * jam_pressure + tracking_pressure, 0.0, 1.0))
        target_dt = max_dt - complexity * (max_dt - min_dt)
        prev_dt = getattr(self, "current_decision_dt", max_dt)
        self.current_decision_dt = c.dynamic_dt_smoothing * prev_dt + (1.0 - c.dynamic_dt_smoothing) * target_dt
        self.dynamic_dt_complexity = complexity

    def _obstacle_pressure(self) -> float:
        if not hasattr(self, "obstacles") or len(self.obstacles) == 0:
            return 0.0
        active_positions = self.positions[~getattr(self, "disabled_uavs", np.zeros(self.cfg.n_uavs, dtype=bool))]
        if len(active_positions) == 0:
            return 0.0
        radius = max(self.cfg.obstacle_radius + self.cfg.safe_radius * self.cfg.obstacle_dt_radius_factor, 1e-6)
        d = np.linalg.norm(active_positions[:, None, :].astype(float) - self.obstacles[None, :, :], axis=2)
        nearest = np.min(d, axis=1)
        return float(np.mean(np.maximum(0.0, 1.0 - nearest / radius)))

    def _pair_pressure(self) -> float:
        if self.cfg.n_uavs <= 1:
            return 0.0
        radius = max(self.cfg.safe_radius * self.cfg.pair_dt_radius_factor, 1e-6)
        pressures = []
        for i in range(self.cfg.n_uavs):
            if getattr(self, "disabled_uavs", np.zeros(self.cfg.n_uavs, dtype=bool))[i]:
                continue
            for j in range(i):
                if getattr(self, "disabled_uavs", np.zeros(self.cfg.n_uavs, dtype=bool))[j]:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                pressures.append(max(0.0, 1.0 - d / radius))
        return float(np.mean(pressures)) if pressures else 0.0

    def action_mask(self, i: int) -> np.ndarray:
        c = self.cfg
        base_mask = super().action_mask(i)
        mask = base_mask.copy()
        if hasattr(self, "disabled_uavs") and self.disabled_uavs[i]:
            inactive = np.zeros(9, dtype=bool)
            inactive[8] = True
            return inactive
        if not hasattr(self, "obstacles"):
            return mask
        if self.cfg.avoidance_mode == "none":
            return mask
        for a in np.flatnonzero(mask[:8]):
            pos = self.positions[i] + self.uav_step_vector(int(a))
            obstacle_risk = self._collides_static(pos) or self._collides_predicted_obstacle(pos)
            pair_risk = self.cfg.avoidance_mode == "dwa_orca" and self._too_close_to_agent(i, pos)
            if obstacle_risk or pair_risk:
                mask[a] = False
        if c.allow_hover and self._should_allow_tracking_hover(i):
            mask[8] = True
        if not mask[:8].any():
            if self.cfg.avoidance_mode != "none" or c.allow_hover:
                mask[8] = True
            else:
                candidates = np.flatnonzero(base_mask[:8])
                if len(candidates):
                    best_action = max(candidates, key=lambda a: self._predicted_clearance(self.positions[i] + self.uav_step_vector(int(a)), i))
                    mask[int(best_action)] = True
                    mask[8] = False
                else:
                    mask[8] = True
        else:
            mask[8] = bool(c.allow_hover and self._should_allow_tracking_hover(i))
        return mask

    def nominal_action_mask(self, i: int) -> np.ndarray:
        """Mode-independent task mask: only dynamics, boundary and turn limits."""
        if hasattr(self, "disabled_uavs") and self.disabled_uavs[i]:
            mask = np.zeros(9, dtype=bool)
            mask[8] = True
            return mask
        return super().action_mask(i).copy()

    def _next_position(self, i: int, action: int) -> np.ndarray:
        return self.positions[i] if action == 8 else self.positions[i] + self.uav_step_vector(int(action))

    def _safety_filtered_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=int).copy()
        proposed = self.positions.copy()
        for i, action in enumerate(actions):
            if self.disabled_uavs[i]:
                actions[i] = 8
                proposed[i] = self.positions[i]
                continue
            base_mask = self.nominal_action_mask(i)
            if action < 0 or action >= 9 or not base_mask[action]:
                actions[i] = 8
                proposed[i] = self.positions[i]
                continue
            if action == 8:
                proposed[i] = self.positions[i]
                continue
            candidate = self._next_position(i, int(action))
            if self.cfg.avoidance_mode != "none" and (
                self._collides_static(candidate) or self._collides_predicted_obstacle(candidate)
            ):
                actions[i] = self._safest_action(i, proposed, avoid_pairs=False)
                candidate = self._next_position(i, int(actions[i]))
            proposed[i] = candidate

        if self.cfg.avoidance_mode != "dwa_orca":
            return actions

        guard_radius = self.cfg.safe_radius + self.cfg.pair_prediction_buffer
        for _ in range(max(1, self.cfg.n_uavs * 2)):
            changed = False
            for i in range(self.cfg.n_uavs):
                if self.disabled_uavs[i]:
                    continue
                for j in range(i):
                    if self.disabled_uavs[j]:
                        continue
                    too_close = np.linalg.norm(proposed[i] - proposed[j]) < guard_radius
                    swap_risk = (
                        actions[i] != 8
                        and actions[j] != 8
                        and np.linalg.norm(proposed[i] - self.positions[j]) <= self.cfg.uav_collision_radius
                        and np.linalg.norm(proposed[j] - self.positions[i]) <= self.cfg.uav_collision_radius
                    )
                    if not (too_close or swap_risk):
                        continue
                    loser = i if self._predicted_clearance(proposed[i], i) <= self._predicted_clearance(proposed[j], j) else j
                    replacement = self._safest_action(loser, proposed, avoid_pairs=True)
                    new_pos = self._next_position(loser, replacement)
                    if actions[loser] != replacement or np.linalg.norm(proposed[loser] - new_pos) > 1e-9:
                        actions[loser] = replacement
                        proposed[loser] = new_pos
                        changed = True
            if not changed:
                break
        return actions

    def _safest_action(self, i: int, proposed: np.ndarray, avoid_pairs: bool = True) -> int:
        mask = self.nominal_action_mask(i)
        mask[8] = True
        best_action = 8
        best_score = -float("inf")
        for action in np.flatnonzero(mask):
            pos = self._next_position(i, int(action))
            if action != 8 and (self._collides_static(pos) or self._collides_predicted_obstacle(pos)):
                continue
            clearance = self._predicted_clearance(pos, i)
            pair_clearance = float(self.cfg.grid_size)
            for j in range(self.cfg.n_uavs):
                if j != i and not self.disabled_uavs[j]:
                    pair_clearance = min(pair_clearance, float(np.linalg.norm(proposed[j] - pos)))
            hover_bonus = 0.25 if action == 8 and pair_clearance < self.cfg.safe_radius * 1.5 else 0.0
            pair_weight = 0.7 if avoid_pairs else 0.0
            score = 2.0 * clearance + pair_weight * pair_clearance + hover_bonus - (0.03 if action == 8 else 0.0)
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action

    def step_search(self, actions, communication_success: np.ndarray | None = None) -> dict:
        c = self.cfg
        self.local_belief_age_map = np.minimum(
            self.local_belief_age_map + 1.0, c.max_belief_age
        )
        active_uav_exposure = int(np.count_nonzero(~self.disabled_uavs))
        prev_uncertainty = self.uncertainty()
        prev_coverage = self.coverage()
        prev_positions = self.positions.copy()
        prev_found = self.found_targets.copy()
        prev_tracked = self.tracked_targets.copy()
        discovery_only = c.target_completion_steps <= 0
        undiscovered_before = ~prev_found
        prev_target_dist = self._mean_target_distance(prev_positions, undiscovered_before)
        raw_actions = np.asarray(list(actions), dtype=int)
        actions = self._safety_filtered_actions(raw_actions)
        safety_changed = actions != raw_actions
        safety_interventions = int(np.count_nonzero(actions != raw_actions))
        invalid_action_count = 0
        disabled_action_count = 0
        for i, a in enumerate(actions):
            if self.disabled_uavs[i]:
                disabled_action_count += int(a != 8)
                continue
            if a == 8:
                continue
            if self.action_mask(i)[a]:
                self.positions[i] = self._next_position(i, int(a))
                self.headings[i] = a
            else:
                invalid_action_count += 1
        pair_conflicts = self._count_pair_conflicts()
        uav_collisions, pair_crash_ids = self._count_uav_collisions()
        obstacle_conflicts, obstacle_crash_ids = self._count_obstacle_conflicts()
        collision_count = obstacle_conflicts + uav_collisions
        newly_crashed = self._apply_crashes(pair_crash_ids | obstacle_crash_ids)
        movement = np.linalg.norm(self.positions - prev_positions, axis=1)
        step_path_length = float(np.sum(movement))
        self.agent_distance_travelled += movement
        revisit_gain = self._update_local_revisit_maps(movement)
        new_cells = 0
        for i, pos in enumerate(self.positions):
            if self.disabled_uavs[i]:
                continue
            x, y = np.clip(pos.astype(int), 0, c.grid_size - 1)
            new_cells += int(self.visit_counts[y, x] == 0)
            self.visit_counts[y, x] += 1
        for i in range(c.n_uavs):
            if not self.disabled_uavs[i]:
                self._scan_and_update(i)
        tracking_metrics = self._update_tracking(prev_found, prev_tracked)
        stagnation_penalty, stagnant_agents = self._update_stagnation(movement, safety_changed)
        # Compare against the same pre-step target set.  This avoids an
        # artificial distance jump when a newly detected target drops out of
        # the undiscovered set.
        target_progress = prev_target_dist - self._mean_target_distance(self.positions, undiscovered_before)
        self._outage_graph_fusion()
        self._move_dynamic_entities()
        self.t += 1
        self._update_outage_state()
        self._update_dynamic_dt()
        search_reward = prev_uncertainty - self.uncertainty()
        coverage_gain = max(0.0, self.coverage() - prev_coverage)
        phase_search_weight = 1.0 if self.task_phase == "search" else 0.35
        phase_tracking_bonus = 1.0 if self.task_phase == "tracking" else 0.45
        safety_penalty = (
            c.invalid_action_penalty * invalid_action_count
            + c.disabled_action_penalty * disabled_action_count
            + c.pair_penalty * pair_conflicts
            + c.obstacle_penalty * obstacle_conflicts
            + c.uav_collision_penalty * uav_collisions
            + c.crash_penalty * len(newly_crashed)
            + stagnation_penalty
        )
        previous_discovery_rate = float(np.mean(prev_found))
        discovery_rate = tracking_metrics["discovery_rate"]
        discovery_milestone = c.discovery_milestone_reward * c.n_targets * (
            discovery_rate**2 - previous_discovery_rate**2
        )
        discovery_terminal = c.all_targets_discovered_reward if discovery_rate >= 1.0 and previous_discovery_rate < 1.0 else 0.0
        if discovery_only:
            if c.reward_mode == "team_potential":
                # Discovery-only objective with deliberately minimal shaping.
                #
                # D(s) is the fraction of targets found by the team and C(s)
                # is the union-map coverage.  Their increments reward outcomes,
                # not hand-coded actions.  Communication is therefore valuable
                # only when the learned policy uses it to improve team search;
                # link density, message count, distance-to-target, revisits and
                # intermediate milestones receive no direct reward.
                tail_exponent = max(c.discovery_tail_exponent, 1.0)
                discovery_gain = max(
                    0.0,
                    discovery_rate**tail_exponent
                    - previous_discovery_rate**tail_exponent,
                )
                safety_event_rate = float(
                    (invalid_action_count + collision_count) / max(active_uav_exposure, 1)
                )
                reward = (
                    c.discovery_potential_weight * discovery_gain
                    + c.exploration_potential_weight * coverage_gain
                    - c.safety_cost_weight * safety_event_rate
                )
            elif c.reward_mode == "legacy_shaped":
                reward = (
                    search_reward
                    + c.coverage_gain_reward * coverage_gain
                    + c.new_cell_reward * new_cells
                    + c.revisit_gain_reward * revisit_gain
                    + c.discovery_reward * tracking_metrics["new_discoveries"]
                    + discovery_milestone
                    + discovery_terminal
                    + c.discovery_progress_reward * float(np.clip(target_progress, -1.0, 1.0))
                    - c.search_step_penalty
                    - safety_penalty
                )
            else:
                raise ValueError(
                    f"Unknown reward_mode={c.reward_mode!r}; "
                    "expected 'team_potential' or 'legacy_shaped'."
                )
        else:
            reward = (
                phase_search_weight * search_reward
                + phase_search_weight * c.coverage_gain_reward * coverage_gain
                + phase_search_weight * c.new_cell_reward * new_cells
                + phase_search_weight * c.revisit_gain_reward * revisit_gain
                + c.discovery_reward * tracking_metrics["new_discoveries"]
                + phase_tracking_bonus * c.tracking_reward * tracking_metrics["tracked_count"]
                + c.persistent_tracking_reward * tracking_metrics["persistent_tracks"]
                + c.tracking_streak_reward * tracking_metrics["streak_gain"]
                - c.tracking_streak_loss_penalty * tracking_metrics["streak_loss"]
                + c.completion_reward * tracking_metrics["new_completions"]
                + c.escort_reward * tracking_metrics["escort_score"]
                + c.target_progress_reward * float(np.clip(target_progress, -1.0, 1.0))
                - c.target_lost_penalty * tracking_metrics["lost_tracks"]
                - safety_penalty
            )
        legacy_components_active = (not discovery_only) or c.reward_mode == "legacy_shaped"
        terminated = discovery_rate >= 1.0 if discovery_only else tracking_metrics["completion_rate"] >= 1.0
        truncated = self.t >= c.search_steps and not terminated
        return {
            "obs": self.observe_search(),
            "reward": float(reward),
            "coverage": self.coverage(),
            "target_discovery_rate": tracking_metrics["discovery_rate"],
            "target_tracking_rate": tracking_metrics["tracking_rate"],
            "target_completion_rate": tracking_metrics["completion_rate"],
            "task_phase": self.task_phase,
            "lost_tracks": tracking_metrics["lost_tracks"],
            "escort_score": tracking_metrics["escort_score"],
            "target_progress": float(target_progress),
            "collisions": int(collision_count),
            "new_crashes": int(len(newly_crashed)),
            "crashed_uavs": self.crashed_uavs.copy().astype(int).tolist(),
            "newly_crashed_uavs": sorted(int(i) for i in newly_crashed),
            "invalid_actions": int(invalid_action_count),
            "disabled_actions": int(disabled_action_count),
            "safety_interventions": safety_interventions,
            "safety_intervention_rate": float(safety_interventions / max(c.n_uavs, 1)),
            "coverage_gain": float(coverage_gain),
            "new_cells": int(new_cells),
            "revisit_gain": float(revisit_gain),
            "stagnant_agents": int(stagnant_agents),
            "stagnation_penalty": float(stagnation_penalty),
            "uav_collisions": int(uav_collisions),
            "pair_conflicts": int(pair_conflicts),
            "obstacle_conflicts": int(obstacle_conflicts),
            "graph_density": float((self.last_graph > 0).sum() / max(c.n_uavs * (c.n_uavs - 1), 1)),
            "outage_rate": float(np.mean(self.disconnected_uavs)),
            "disabled_rate": float(np.mean(self.disabled_uavs)),
            "active_uav_count": int(np.count_nonzero(~self.disabled_uavs)),
            "active_uav_exposure": active_uav_exposure,
            "step_path_length": step_path_length,
            "mean_interference": float(np.mean(self.interference_level)),
            "mean_cell_belief_age": float(np.mean(self.local_belief_age_map)),
            "p95_cell_belief_age": float(
                np.percentile(self.local_belief_age_map, 95)
            ),
            "current_decision_dt": float(self.current_decision_dt),
            "dynamic_dt_complexity": float(self.dynamic_dt_complexity),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated or truncated),
            "raw_actions": raw_actions.astype(int).tolist(),
            "executed_actions": actions.astype(int).tolist(),
            "action_agreement_rate": float(np.mean(actions == raw_actions)),
            "reward_components": {
                "team_discovery_potential": float(
                    c.discovery_potential_weight
                    * max(
                        0.0,
                        discovery_rate ** max(c.discovery_tail_exponent, 1.0)
                        - previous_discovery_rate
                        ** max(c.discovery_tail_exponent, 1.0),
                    )
                    if discovery_only and c.reward_mode == "team_potential"
                    else 0.0
                ),
                "team_exploration_potential": float(
                    c.exploration_potential_weight * coverage_gain
                    if discovery_only and c.reward_mode == "team_potential"
                    else 0.0
                ),
                "team_safety_cost": float(
                    -c.safety_cost_weight
                    * (invalid_action_count + collision_count)
                    / max(active_uav_exposure, 1)
                    if discovery_only and c.reward_mode == "team_potential"
                    else 0.0
                ),
                "search": float(phase_search_weight * search_reward if legacy_components_active else 0.0),
                "coverage": float(phase_search_weight * c.coverage_gain_reward * coverage_gain if legacy_components_active else 0.0),
                "new_cells": float(phase_search_weight * c.new_cell_reward * new_cells if legacy_components_active else 0.0),
                "revisit": float(phase_search_weight * c.revisit_gain_reward * revisit_gain if legacy_components_active else 0.0),
                "discovery": float(c.discovery_reward * tracking_metrics["new_discoveries"] if legacy_components_active else 0.0),
                "discovery_milestone": float(discovery_milestone if discovery_only and c.reward_mode == "legacy_shaped" else 0.0),
                "discovery_terminal": float(discovery_terminal if discovery_only and c.reward_mode == "legacy_shaped" else 0.0),
                "discovery_progress": float(c.discovery_progress_reward * np.clip(target_progress, -1.0, 1.0) if discovery_only and c.reward_mode == "legacy_shaped" else 0.0),
                "step_penalty": float(-c.search_step_penalty if discovery_only and c.reward_mode == "legacy_shaped" else 0.0),
                "tracking": float(phase_tracking_bonus * c.tracking_reward * tracking_metrics["tracked_count"]),
                "persistent_tracking": float(c.persistent_tracking_reward * tracking_metrics["persistent_tracks"]),
                "tracking_progress": float(c.tracking_streak_reward * tracking_metrics["streak_gain"]),
                "tracking_regression": float(-c.tracking_streak_loss_penalty * tracking_metrics["streak_loss"]),
                "completion": float(c.completion_reward * tracking_metrics["new_completions"]),
                "lost_track": float(-c.target_lost_penalty * tracking_metrics["lost_tracks"]),
                "escort": float(c.escort_reward * tracking_metrics["escort_score"]),
                "lost_target_penalty": float(-c.target_lost_penalty * tracking_metrics["lost_tracks"]),
                "safety_penalty": float(
                    -c.invalid_action_penalty * invalid_action_count
                    - c.pair_penalty * pair_conflicts
                    - c.obstacle_penalty * obstacle_conflicts
                    - c.uav_collision_penalty * uav_collisions
                    if legacy_components_active
                    else 0.0
                ),
                "crash_penalty": float(
                    -c.crash_penalty * len(newly_crashed)
                    if legacy_components_active
                    else 0.0
                ),
                "stagnation_penalty": float(
                    -stagnation_penalty if legacy_components_active else 0.0
                ),
            },
        }

    def _update_stagnation(self, movement: np.ndarray, safety_changed: np.ndarray) -> tuple[float, int]:
        """Penalize avoidable per-agent idling, not crashes, tracking, or safety overrides."""
        penalty = 0.0
        stagnant_agents = 0
        for i in range(self.cfg.n_uavs):
            exempt = self.disabled_uavs[i] or self._is_productive_tracking_hold(i) or bool(safety_changed[i]) or not self.action_mask(i)[:8].any()
            if exempt or movement[i] >= 0.1:
                self.stagnation_steps[i] = 0
                continue
            self.stagnation_steps[i] += 1
            if self.stagnation_steps[i] >= self.cfg.stagnation_threshold:
                stagnant_agents += 1
                penalty += self.cfg.stagnation_penalty
            if self.stagnation_steps[i] >= self.cfg.long_stagnation_threshold:
                penalty += self.cfg.long_stagnation_penalty
        return float(penalty), stagnant_agents

    def _is_productive_tracking_hold(self, i: int) -> bool:
        if self.cfg.target_completion_steps <= 0:
            return False
        active_found = self.found_targets & ~self.completed_targets
        if not np.any(active_found):
            return False
        distance = np.linalg.norm(self.dynamic_targets[active_found] - self.positions[i].astype(float), axis=1)
        return bool(np.min(distance) <= self.cfg.tracking_radius)

    def _should_allow_tracking_hover(self, i: int) -> bool:
        if not hasattr(self, "dynamic_targets"):
            return False
        active = ~self.found_targets if self.cfg.target_completion_steps <= 0 else ~self.completed_targets
        if not np.any(active):
            return False
        d = np.linalg.norm(self.dynamic_targets[active] - self.positions[i].astype(float), axis=1)
        near_tracking_zone = bool(np.min(d) <= self.cfg.tracking_radius * 1.15)
        return self.task_phase == "tracking" or near_tracking_zone

    def _mean_active_target_distance(self, positions: np.ndarray) -> float:
        if not hasattr(self, "dynamic_targets"):
            return 0.0
        active = ~self.completed_targets
        if not np.any(active):
            return 0.0
        working = ~getattr(self, "disabled_uavs", np.zeros(self.cfg.n_uavs, dtype=bool))
        working_positions = positions[working]
        if len(working_positions) == 0:
            return float(self.cfg.grid_size)
        d = np.linalg.norm(working_positions[:, None, :].astype(float) - self.dynamic_targets[None, active, :], axis=2)
        return float(np.mean(np.min(d, axis=0)))

    def _mean_target_distance(self, positions: np.ndarray, target_mask: np.ndarray) -> float:
        """Mean nearest-team distance for an explicit, fixed target set."""
        target_mask = np.asarray(target_mask, dtype=bool)
        if not hasattr(self, "dynamic_targets") or not np.any(target_mask):
            return 0.0
        working = ~getattr(self, "disabled_uavs", np.zeros(self.cfg.n_uavs, dtype=bool))
        working_positions = positions[working]
        if len(working_positions) == 0:
            return float(self.cfg.grid_size)
        distance = np.linalg.norm(
            working_positions[:, None, :].astype(float) - self.dynamic_targets[None, target_mask, :], axis=2
        )
        return float(np.mean(np.min(distance, axis=0)))

    def _scan_and_update(self, i: int) -> None:
        c = self.cfg
        x0, y0 = self.positions[i].astype(float)
        hit_delta = math.log(c.pf / c.pd)
        miss_delta = math.log((1.0 - c.pf) / (1.0 - c.pd))
        for y in range(max(0, int(y0) - 1), min(c.grid_size, int(y0) + 2)):
            for x in range(max(0, int(x0) - 1), min(c.grid_size, int(x0) + 2)):
                cell_center = np.array([x, y], dtype=float)
                target_dist = np.linalg.norm(self.dynamic_targets - cell_center, axis=1)
                target = bool(np.any(target_dist <= 0.6))
                detected = self.rng.random() < (c.pd if target else c.pf)
                self.local_q[i, y, x] += hit_delta if detected else miss_delta
                self.local_belief_age_map[i, y, x] = 0.0
                if detected and target:
                    self.found_targets[target_dist <= c.sensor_radius] = True

    def _mark_revisit_footprint_fresh(self, i: int) -> None:
        """Reset only UAV i's locally sensed cells in the revisit layer."""
        if self.disabled_uavs[i]:
            return
        c = self.cfg
        x0, y0 = self.positions[i].astype(float)
        radius = int(math.ceil(c.sensor_radius))
        for y in range(max(0, int(y0) - radius), min(c.grid_size, int(y0) + radius + 1)):
            for x in range(max(0, int(x0) - radius), min(c.grid_size, int(x0) + radius + 1)):
                if np.linalg.norm(np.array([x, y], dtype=float) - np.array([x0, y0])) <= c.sensor_radius:
                    self.local_revisit_map[i, y, x] = 0.0

    def _update_local_revisit_maps(self, movement: np.ndarray) -> float:
        """Age, diffuse and locally reset the third search-map layer.

        A revisit reward is emitted only for a stale cell that had already
        been visited before this step. Merely entering a never-seen cell is
        handled by the separate new-cell/coverage rewards.
        """
        c = self.cfg
        if not c.revisit_map_enabled:
            self.local_revisit_map.fill(0.0)
            return 0.0
        rate = float(np.clip(c.revisit_aging_rate, 0.0, 1.0))
        diffusion = float(np.clip(c.revisit_diffusion, 0.0, 1.0))
        maps = self.local_revisit_map
        maps += rate * (1.0 - maps)
        if diffusion > 0.0:
            padded = np.pad(maps, ((0, 0), (1, 1), (1, 1)), mode="edge")
            neighbor_mean = (
                padded[:, :-2, 1:-1]
                + padded[:, 2:, 1:-1]
                + padded[:, 1:-1, :-2]
                + padded[:, 1:-1, 2:]
            ) / 4.0
            maps[:] = (1.0 - diffusion) * maps + diffusion * neighbor_mean
        revisit_gain = 0.0
        for i in range(c.n_uavs):
            if self.disabled_uavs[i]:
                continue
            x, y = np.clip(self.positions[i].astype(int), 0, c.grid_size - 1)
            stale_value = float(maps[i, y, x])
            was_previously_visited = bool(self.visit_counts[y, x] > 0)
            if movement[i] >= 0.1 and was_previously_visited and stale_value >= c.revisit_reward_threshold:
                revisit_gain += stale_value
            self._mark_revisit_footprint_fresh(i)
        np.clip(maps, 0.0, 1.0, out=maps)
        return revisit_gain

    def _update_tracking(self, prev_found: np.ndarray, prev_tracked: np.ndarray) -> dict:
        c = self.cfg
        active_positions = self.positions[~self.disabled_uavs]
        if len(active_positions) == 0:
            d = np.full((1, self.cfg.n_targets), self.cfg.grid_size, dtype=float)
        else:
            d = np.linalg.norm(active_positions[:, None, :].astype(float) - self.dynamic_targets[None, :, :], axis=2)
        observed = np.min(d, axis=0) <= c.sensor_radius
        tracked = np.min(d, axis=0) <= c.tracking_radius
        self.found_targets |= observed
        newly_found_mask = self.found_targets & ~prev_found
        self.target_first_discovery_step[newly_found_mask] = self.t + 1
        if c.target_completion_steps <= 0:
            # Zero is a real discovery-only mode, not a zero-length tracking
            # streak (which would otherwise complete every target instantly).
            self.tracked_targets.fill(False)
            self.target_tracking_streak.fill(0)
            self.target_tracking_miss_count.fill(0)
            self.task_phase = "search"
            discovery_rate = float(np.mean(self.found_targets))
            return {
                "new_discoveries": int(np.count_nonzero(self.found_targets & ~prev_found)),
                "tracked_count": 0,
                "persistent_tracks": 0,
                "streak_gain": 0.0,
                "streak_loss": 0.0,
                "new_completions": 0,
                "lost_tracks": 0,
                "escort_score": 0.0,
                "discovery_rate": discovery_rate,
                "tracking_rate": 0.0,
                # In discovery-only mode task completion means all targets
                # have been found; keeping this alias preserves old loggers.
                "completion_rate": discovery_rate,
            }
        active = ~self.completed_targets
        previous_streak = self.target_tracking_streak.copy()
        self.tracked_targets = self.found_targets & tracked & active
        self.target_tracking_miss_count[self.tracked_targets] = 0
        self.target_tracking_streak[self.tracked_targets] += 1
        missed = active & ~self.tracked_targets & (self.target_tracking_streak > 0)
        self.target_tracking_miss_count[missed] += 1
        within_grace = missed & (self.target_tracking_miss_count <= c.tracking_grace_steps)
        self.target_tracking_streak[within_grace] = np.maximum(
            0,
            self.target_tracking_streak[within_grace] - max(int(c.tracking_streak_decay), 0),
        )
        confirmed_lost = missed & (self.target_tracking_miss_count > c.tracking_grace_steps)
        self.target_tracking_streak[confirmed_lost] = 0
        never_tracking = active & ~self.tracked_targets & (previous_streak <= 0)
        self.target_tracking_miss_count[never_tracking] = 0
        newly_completed = active & (self.target_tracking_streak >= c.target_completion_steps)
        new_completion_count = int(np.count_nonzero(newly_completed))
        if np.any(newly_completed):
            self.completed_targets |= newly_completed
            self.target_velocity[newly_completed] = 0.0
            self.tracked_targets[newly_completed] = False
        active_found = self.found_targets & ~self.completed_targets
        if float(np.mean(active_found)) >= c.tracking_stage_threshold or np.any(self.tracked_targets):
            self.task_phase = "tracking"
        else:
            self.task_phase = "search"
        new_discoveries = int(np.count_nonzero(self.found_targets & ~prev_found))
        persistent_tracks = int(np.count_nonzero(self.tracked_targets & prev_tracked))
        lost_tracks = int(np.count_nonzero(confirmed_lost & ~self.completed_targets))
        streak_delta = self.target_tracking_streak.astype(float) - previous_streak.astype(float)
        streak_gain = float(np.sum(np.maximum(streak_delta[active], 0.0)))
        streak_loss = float(np.sum(np.maximum(-streak_delta[active], 0.0)))
        escort_score = self._escort_score(d)
        self.prev_tracked_targets = self.tracked_targets.copy()
        return {
            "new_discoveries": new_discoveries,
            "tracked_count": int(np.count_nonzero(self.tracked_targets)),
            "persistent_tracks": persistent_tracks,
            "streak_gain": streak_gain,
            "streak_loss": streak_loss,
            "new_completions": new_completion_count,
            "lost_tracks": lost_tracks,
            "escort_score": escort_score,
            "discovery_rate": float(np.mean(self.found_targets)),
            "tracking_rate": float(np.mean(self.tracked_targets | self.completed_targets)),
            "completion_rate": float(np.mean(self.completed_targets)),
        }

    def _escort_score(self, distance_matrix: np.ndarray) -> float:
        active_found = self.found_targets & ~self.completed_targets
        if not np.any(active_found):
            return 0.0
        score = 0.0
        found_indices = np.flatnonzero(active_found)
        for target_idx in found_indices:
            distances = np.sort(distance_matrix[:, target_idx])
            nearest = distances[: min(2, len(distances))]
            score += float(np.mean(np.maximum(0.0, 1.0 - np.abs(nearest - self.cfg.tracking_radius) / self.cfg.tracking_radius)))
        return score / max(len(found_indices), 1)

    def _move_dynamic_entities(self) -> None:
        if self.cfg.dynamic_targets_enabled:
            active_targets = ~self.completed_targets
            moved, velocity = self._move_points(self.dynamic_targets[active_targets], self.target_velocity[active_targets])
            self.dynamic_targets[active_targets] = moved
            self.target_velocity[active_targets] = velocity
        if self.cfg.dynamic_obstacles_enabled:
            self.obstacles, self.obstacle_velocity = self._move_points(self.obstacles, self.obstacle_velocity)
        self._separate_obstacles_from_targets()
        self.target_grid = self._target_occupancy()

    def _move_points(self, points: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) == 0:
            return points, velocity
        c = self.cfg
        dt = getattr(self, "current_decision_dt", self.cfg.decision_dt)
        points = points + velocity * dt
        for idx in range(len(points)):
            for axis in range(2):
                if points[idx, axis] < 0.5 or points[idx, axis] > c.grid_size - 1.5:
                    velocity[idx, axis] *= -1.0
            points[idx] = np.clip(points[idx], 0.5, c.grid_size - 1.5)
        return points, velocity

    def _separate_obstacles_from_targets(self) -> None:
        if len(self.obstacles) == 0 or len(self.dynamic_targets) == 0:
            return
        for i in range(len(self.obstacles)):
            d = np.linalg.norm(self.dynamic_targets - self.obstacles[i], axis=1)
            if np.min(d) < self.cfg.obstacle_radius + 0.4:
                self.obstacle_velocity[i] *= -1.0

    def _target_occupancy(self) -> np.ndarray:
        c = self.cfg
        grid = np.zeros((c.grid_size, c.grid_size), dtype=bool)
        cells = np.rint(self.dynamic_targets).astype(int)
        cells = np.clip(cells, 0, c.grid_size - 1)
        grid[cells[:, 1], cells[:, 0]] = True
        return grid

    def step_joint(self, search_actions: np.ndarray) -> dict:
        outage = self._communication_status()
        result = self.step_search(search_actions)
        result["outage_success_rate"] = float(np.mean(outage["success"]))
        result["outage_rate"] = float(np.mean(self.disconnected_uavs))
        result["disabled_rate"] = float(np.mean(self.disabled_uavs))
        result["active_uav_count"] = int(np.count_nonzero(~self.disabled_uavs))
        result["mean_interference"] = float(np.mean(self.interference_level))
        result["disabled_uavs"] = self.disabled_uavs.copy().astype(int).tolist()
        result["disconnected_uavs"] = self.disconnected_uavs.copy().astype(int).tolist()
        result["communication_success_rate"] = result["outage_success_rate"]
        result["communication_packet_loss_rate"] = result["outage_rate"]
        result["communication_latency_ms"] = float(np.mean(outage["latency_ms"]))
        result["communication_throughput_kbps"] = float(np.mean(outage["throughput_kbps"]))
        result["communication_logs"] = self._communication_slot_log(outage)
        result["spectrum_reward"] = 0.0
        result["spectrum_success_rate"] = result["communication_success_rate"]
        result["spectrum_packet_loss_rate"] = result["communication_packet_loss_rate"]
        result["spectrum_latency_ms"] = result["communication_latency_ms"]
        result["spectrum_throughput_kbps"] = result["communication_throughput_kbps"]
        result["spectrum_slot_logs"] = result["communication_logs"]
        return result

    def _communication_status(self) -> dict:
        success = (~self.disconnected_uavs) & (~self.disabled_uavs)
        packet_loss = np.where(success, 0.0, 1.0)
        latency_ms = np.where(success, self.cfg.base_latency_ms, self.cfg.timeout_latency_ms)
        throughput_kbps = np.where(success, self.cfg.max_throughput_kbps, 0.0)
        return {
            "success": success,
            "packet_loss": packet_loss,
            "latency_ms": latency_ms,
            "throughput_kbps": throughput_kbps,
            "jammed": [],
        }

    def _communication_slot_log(self, comm: dict) -> list[list[dict]]:
        rows = []
        for i in range(self.cfg.n_uavs):
            rows.append(
                {
                    "slot": 0,
                    "uav_id": i,
                    "success": bool(comm["success"][i]),
                    "packet_loss": float(comm["packet_loss"][i]),
                    "latency_ms": float(comm["latency_ms"][i]),
                    "throughput_kbps": float(comm["throughput_kbps"][i]),
                }
            )
        return [rows]

    def _update_outage_state(self) -> None:
        c = self.cfg
        if c.comm_mode == "full":
            self.interference_level = np.zeros(c.n_uavs, dtype=float)
            self.disconnected_uavs = self.crashed_uavs.copy()
            self.disabled_uavs = self.crashed_uavs.copy()
            self.spectrum_t += 1
            return
        centers = self._jammer_positions()
        if len(centers) == 0:
            level = np.zeros(c.n_uavs, dtype=float)
        else:
            d = np.linalg.norm(self.positions[:, None, :].astype(float) - centers[None, :, :], axis=2)
            pressure = np.maximum(0.0, 1.0 - d / max(c.jammer_effect_radius, 1e-6))
            level = np.clip(np.max(pressure, axis=1), 0.0, 1.0)
        self.interference_level = level
        outage_prob = np.clip(c.outage_base_prob + c.outage_jammed_prob * level, 0.0, 0.98)
        recovery_prob = np.clip(c.outage_recovery_prob * (1.0 - level), 0.0, 1.0)
        new_outage = self.disconnected_uavs.copy()
        recover = self.rng.random(c.n_uavs) < recovery_prob
        fail_link = self.rng.random(c.n_uavs) < outage_prob
        new_outage = np.where(new_outage, ~recover, fail_link)
        new_outage[self.crashed_uavs] = True
        failure_prob = np.clip(c.failure_base_prob + c.failure_jammed_prob * level, 0.0, 0.75)
        repair = self.rng.random(c.n_uavs) < c.failure_recovery_prob
        temporary_disabled = (self.disabled_uavs & ~self.crashed_uavs & ~repair) | (
            self.rng.random(c.n_uavs) < failure_prob
        )
        new_disabled = self.crashed_uavs | temporary_disabled
        self.disabled_uavs = new_disabled.astype(bool)
        self.disconnected_uavs = (new_outage | self.disabled_uavs).astype(bool)
        self.spectrum_t += 1

    def _jammer_positions(self) -> np.ndarray:
        c = self.cfg
        if c.n_jammers <= 0:
            return np.empty((0, 2), dtype=float)
        phase = (getattr(self, "spectrum_t", 0) + np.arange(c.n_jammers) * max(c.grid_size // max(c.n_jammers, 1), 1)) % c.grid_size
        y = (0.5 * c.grid_size * (1.0 + np.sin((phase + self.jammer_phase[: c.n_jammers]) * 2.0 * math.pi / max(c.grid_size, 1))))
        x = phase.astype(float)
        return np.column_stack([x, np.clip(y, 0.0, c.grid_size - 1.0)])

    def _outage_graph_fusion(self) -> None:
        c = self.cfg
        history_mode = self._history_mode()
        if c.comm_mode == "full":
            active_idx = np.flatnonzero(~self.disabled_uavs)
            if len(active_idx):
                fused_q, fused_age = self._fuse_belief_sources(
                    self.local_q[active_idx],
                    self.local_belief_age_map[active_idx],
                    np.ones(len(active_idx), dtype=float),
                )
                self.local_q[active_idx] = fused_q
                self.local_belief_age_map[active_idx] = fused_age
                self.q_map = fused_q.copy()
                fused_revisit = np.min(self.local_revisit_map[active_idx], axis=0)
                self.local_revisit_map[active_idx] = fused_revisit
            active = (~self.disabled_uavs).astype(float)
            self.last_graph = (active[:, None] * active[None, :]) * (1.0 - np.eye(c.n_uavs))
            self.belief_age[:] = 0.0
            return
        if c.comm_mode == "none":
            self.belief_age += 1.0
            self.last_graph = np.zeros((c.n_uavs, c.n_uavs), dtype=float)
            candidates = np.vstack([self.q_map[None, :, :], self.local_q])
            best = np.argmax(np.abs(candidates), axis=0)
            self.q_map = np.take_along_axis(candidates, best[None, :, :], axis=0)[0]
            return
        graph = self._comm_graph()
        new_local = self.local_q.copy()
        new_age_map = self.local_belief_age_map.copy()
        new_revisit = self.local_revisit_map.copy()
        for i in range(c.n_uavs):
            if self.disabled_uavs[i]:
                self.belief_age[i] += 1.0
                continue
            sources = np.flatnonzero(graph[i] > 0)
            if len(sources) == 0:
                self.belief_age[i] += 1.0
                continue
            candidates = [self.local_q[i]]
            candidate_ages = [self.local_belief_age_map[i]]
            weights = [1.0]
            for j in sources:
                candidates.append(self.local_q[j])
                # A received packet is at least one communication step old.
                candidate_ages.append(
                    np.minimum(self.local_belief_age_map[j] + 1.0, c.max_belief_age)
                )
                weights.append(graph[i, j] if history_mode != "none" else 1.0)
                freshness_penalty = (
                    (1.0 - float(graph[i, j])) * c.revisit_aging_rate
                    if history_mode == "full"
                    else 0.0
                )
                received_revisit = np.clip(
                    self.local_revisit_map[j] + freshness_penalty, 0.0, 1.0
                )
                # Lower values mean fresher observations. Taking the minimum
                # propagates only knowledge actually received over this edge.
                new_revisit[i] = np.minimum(new_revisit[i], received_revisit)
            new_local[i], new_age_map[i] = self._fuse_belief_sources(
                np.stack(candidates), np.stack(candidate_ages), np.asarray(weights)
            )
            self.belief_age[i] = float(np.mean(new_age_map[i]))
        self.local_q = new_local
        self.local_belief_age_map = new_age_map
        self.local_revisit_map = new_revisit
        candidates = np.vstack([self.q_map[None, :, :], self.local_q])
        best = np.argmax(np.abs(candidates), axis=0)
        self.q_map = np.take_along_axis(candidates, best[None, :, :], axis=0)[0]
        self.last_graph = graph
        self.belief_age = self.belief_age * c.age_decay

    def _fuse_belief_sources(
        self, beliefs: np.ndarray, ages: np.ndarray, link_weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse cell beliefs using confidence, link quality and cell-wise AoI.

        ``hard`` reproduces the old winner-takes-all rule for ablations.
        ``soft`` prevents a stale overconfident packet from replacing a recent
        local observation and handles contradictory evidence by interpolation.
        """
        c = self.cfg
        beliefs = np.asarray(beliefs, dtype=float)
        ages = np.asarray(ages, dtype=float)
        link = np.asarray(link_weights, dtype=float)[:, None, None]
        mode = str(getattr(c, "belief_fusion_mode", "soft")).lower()
        if mode == "hard" or not c.freshness_fusion_enabled:
            score = np.abs(beliefs) * link
            best = np.argmax(score, axis=0)
            fused = np.take_along_axis(beliefs, best[None], axis=0)[0]
            fused_age = np.take_along_axis(ages, best[None], axis=0)[0]
            return fused, fused_age.astype(np.float32)
        if mode != "soft":
            raise ValueError(
                f"Unknown belief_fusion_mode={mode!r}; expected 'soft' or 'hard'."
            )

        confidence = c.belief_confidence_floor + (
            1.0 - np.exp(-np.abs(beliefs))
        )
        freshness = np.exp(-ages / max(c.belief_freshness_tau, 1e-6))
        evidence = np.maximum(link * confidence * freshness, 1e-12)
        temperature = max(c.belief_fusion_temperature, 1e-3)
        logits = np.log(evidence) / temperature
        logits -= np.max(logits, axis=0, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(np.sum(weights, axis=0, keepdims=True), 1e-12)
        fused = np.sum(weights * beliefs, axis=0)
        fused_age = np.sum(weights * ages, axis=0)
        return fused, np.clip(fused_age, 0.0, c.max_belief_age).astype(np.float32)

    def _comm_graph(self) -> np.ndarray:
        c = self.cfg
        history_mode = self._history_mode()
        graph = np.zeros((c.n_uavs, c.n_uavs), dtype=float)
        for i in range(c.n_uavs):
            for j in range(c.n_uavs):
                if i == j:
                    continue
                if self.disabled_uavs[i] or self.disabled_uavs[j]:
                    continue
                if self.disconnected_uavs[i] or self.disconnected_uavs[j]:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d > c.comm_radius:
                    continue
                graph[i, j] = (
                    math.exp(-c.graph_beta * d) if history_mode != "none" else 1.0
                )
                if history_mode != "none":
                    graph[i, j] /= 1.0 + self.belief_age[j]
        return graph

    def _history_mode(self) -> str:
        """Return the normalized history ablation mode with legacy compatibility."""
        if not self.cfg.freshness_fusion_enabled:
            return "none"
        mode = str(getattr(self.cfg, "history_mode", "full")).lower()
        if mode not in {"none", "edge_age", "full"}:
            raise ValueError(
                f"Unknown history_mode={mode!r}; expected none, edge_age, or full."
            )
        return mode

    def belief_decisiveness(self) -> float:
        """Paper-defined fraction outside the [theta0, theta1] uncertain band."""
        probability = log_odds_to_prob(self.q_map)
        decisive = (probability <= self.cfg.theta0) | (
            probability >= self.cfg.theta1
        )
        return float(np.mean(decisive))

    def belief_brier_score(self) -> float:
        """Ground-truth Brier score used only for evaluation, never by the actor."""
        probability = log_odds_to_prob(self.q_map)
        truth = np.zeros_like(probability, dtype=float)
        if len(self.dynamic_targets):
            yy, xx = np.indices(probability.shape)
            cells = np.stack([xx, yy], axis=-1).astype(float)
            distances = np.linalg.norm(
                cells[:, :, None, :] - self.dynamic_targets[None, None, :, :],
                axis=-1,
            )
            truth[np.any(distances <= 0.6, axis=-1)] = 1.0
        return float(np.mean((probability - truth) ** 2))

    def graph_features(self) -> np.ndarray:
        c = self.cfg
        out = np.zeros((c.n_uavs, 32), dtype=float)
        for i in range(c.n_uavs):
            pos = self.positions[i].astype(float)
            local_p = log_odds_to_prob(self.local_q[i])
            local_uncertainty = np.exp(-np.abs(self.local_q[i]))
            hotspot = np.array(np.unravel_index(np.argmax(local_uncertainty + 0.1 * local_p), local_p.shape))[::-1]
            if self.disabled_uavs[i]:
                neighbors = []
            elif hasattr(self, "last_graph"):
                neighbors = np.flatnonzero(self.last_graph[i] > 0).tolist()
            else:
                neighbors = []
            if neighbors:
                rel = self.positions[neighbors].astype(float) - pos
                dist = np.linalg.norm(rel, axis=1)
                w = self._softmax(np.exp(-c.graph_beta * dist))
                agent_feature = (w[:, None] * rel).sum(axis=0) / c.grid_size
                agent_count = len(neighbors) / c.n_uavs
            else:
                agent_feature = np.zeros(2)
                agent_count = 0.0
            obstacle_rel, obstacle_dist = self._nearest_obstacle(pos)
            target_rel, target_dist, visible_targets = self._nearest_target_feature(pos, hotspot)
            local_unc = self._local_patch_value(local_uncertainty, pos)
            # Decentralized actors must not receive team-global target status.
            # Derive these features only from targets visible to UAV i or to a
            # neighbor connected in the current communication graph.
            causal_targets = self.causal_target_indices(i)
            active_found = float(len(causal_targets) / max(c.n_targets, 1))
            if len(causal_targets):
                causal_dist = np.linalg.norm(
                    self.dynamic_targets[causal_targets] - pos,
                    axis=1,
                )
                tracked_rate = float(np.count_nonzero(causal_dist <= c.tracking_radius) / max(c.n_targets, 1))
            else:
                tracked_rate = 0.0
            # Reuse the former global-completion slot for a causal local signal:
            # risk that a currently tracked/visible target leaves the tracking
            # radius within the short prediction horizon.
            target_loss_risk = self.local_target_loss_risk(i, causal_targets)
            phase_flag = 1.0 if len(causal_targets) else 0.0
            connected_flag = 0.0 if self.disconnected_uavs[i] else 1.0
            disabled_flag = 1.0 if self.disabled_uavs[i] else 0.0
            frontier_rel, sector_coverage, frontier_value, revisit_rel = self._exploration_feature(
                i, local_p, local_uncertainty
            )
            local_prob = self._local_patch_value(local_p, pos)
            local_revisit = self._local_patch_value(self.local_revisit_map[i], pos)
            local_information_age = self._local_patch_value(
                self.local_belief_age_map[i], pos
            )
            out[i] = np.array(
                [
                    pos[0] / c.grid_size,
                    pos[1] / c.grid_size,
                    math.cos(self.headings[i] * math.pi / 4.0),
                    math.sin(self.headings[i] * math.pi / 4.0),
                    agent_feature[0],
                    agent_feature[1],
                    agent_count,
                    obstacle_rel[0] / c.grid_size,
                    obstacle_rel[1] / c.grid_size,
                    min(obstacle_dist / c.grid_size, 1.0),
                    target_rel[0],
                    target_rel[1],
                    min(target_dist / c.grid_size, 1.0),
                    visible_targets / max(c.n_targets, 1),
                    local_unc,
                    active_found,
                    target_loss_risk,
                    tracked_rate,
                    phase_flag,
                    connected_flag,
                    disabled_flag,
                    min(local_information_age / max(c.belief_freshness_tau, 1e-6), 1.0)
                    if c.freshness_fusion_enabled
                    else 0.0,
                    float(np.clip(self.interference_level[i], 0.0, 1.0)),
                    frontier_rel[0],
                    frontier_rel[1],
                    sector_coverage,
                    frontier_value,
                    min(float(self.stagnation_steps[i]) / max(c.long_stagnation_threshold, 1), 1.0),
                    local_prob,
                    local_revisit,
                    revisit_rel[0],
                    revisit_rel[1],
                ],
                dtype=float,
            )
        return out

    def _exploration_feature(
        self, i: int, target_prob: np.ndarray, uncertainty: np.ndarray
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Return local composite and revisit frontiers without global visit leakage."""
        c = self.cfg
        active = np.flatnonzero(~self.disabled_uavs)
        if i not in active:
            return np.zeros(2), 0.0, 0.0, np.zeros(2)
        rank = int(np.flatnonzero(active == i)[0])
        x_lo = int(rank * c.grid_size / max(len(active), 1))
        x_hi = max(x_lo + 1, int((rank + 1) * c.grid_size / max(len(active), 1)))
        sector_uncertainty = uncertainty[:, x_lo:x_hi]
        sector_prob = target_prob[:, x_lo:x_hi]
        sector_revisit = self.local_revisit_map[i, :, x_lo:x_hi]
        revisit_weight = 0.65 if c.revisit_map_enabled else 0.0
        score = 0.15 * sector_prob + sector_uncertainty + revisit_weight * sector_revisit
        fy, fx = np.unravel_index(int(np.argmax(score)), score.shape)
        frontier = np.array([fx + x_lo, fy], dtype=float)
        rel = (frontier - self.positions[i].astype(float)) / c.grid_size
        value = float(score[fy, fx]) if score.size else 0.0
        if not c.revisit_map_enabled:
            # Keep the uncertainty/probability frontier operational, but mask
            # every revisit-derived channel with a true neutral value.  In
            # particular, never run argmax on an all-zero disabled map because
            # that creates a spurious fixed direction.
            return rel, 0.0, value, np.zeros(2)
        coverage = float(np.mean(sector_revisit < c.revisit_fresh_threshold)) if sector_revisit.size else 0.0
        ry, rx = np.unravel_index(int(np.argmax(sector_revisit)), sector_revisit.shape)
        revisit_frontier = np.array([rx + x_lo, ry], dtype=float)
        revisit_rel = (revisit_frontier - self.positions[i].astype(float)) / c.grid_size
        return rel, coverage, value, revisit_rel

    def agent_observation_vectors(self) -> np.ndarray:
        if not hasattr(self, "obstacles"):
            return np.zeros((self.cfg.n_uavs, self.obs_dim()))
        masks = np.stack([self.action_mask(i) for i in range(self.cfg.n_uavs)]).astype(float)
        return np.concatenate([self.graph_features(), masks], axis=1)

    def causal_target_indices(self, i: int) -> np.ndarray:
        """Targets observable by UAV i under the current communication graph.

        Exact target state is available only when the target lies in UAV i's
        sensor footprint or in the footprint of a currently connected peer.
        Previously found global target truth is deliberately not used.
        """
        if self.disabled_uavs[i] or not len(self.dynamic_targets):
            return np.empty(0, dtype=int)
        observers = [int(i)]
        if hasattr(self, "last_graph"):
            observers.extend(
                int(j)
                for j in np.flatnonzero(self.last_graph[i] > 0)
                if not self.disabled_uavs[int(j)]
            )
        observer_positions = self.positions[np.asarray(sorted(set(observers)), dtype=int)].astype(float)
        distance = np.linalg.norm(
            observer_positions[:, None, :] - self.dynamic_targets[None, :, :],
            axis=2,
        )
        visible = np.any(distance <= self.cfg.sensor_radius, axis=0)
        if self.cfg.target_completion_steps <= 0:
            # Once detected, a target is no longer an actor objective in pure
            # discovery mode.  This prevents lingering near an already found
            # target even though the physical target keeps moving.
            visible &= ~self.found_targets
        else:
            visible &= ~self.completed_targets
        return np.flatnonzero(visible)

    def local_target_loss_risk(self, i: int, target_indices: np.ndarray | None = None) -> float:
        """Short-horizon target-loss risk available to decentralized actors."""
        if target_indices is None:
            target_indices = self.causal_target_indices(i)
        target_indices = np.asarray(target_indices, dtype=int)
        if self.disabled_uavs[i] or target_indices.size == 0:
            return 0.0
        pos = self.positions[i].astype(float)
        current_distance = np.linalg.norm(self.dynamic_targets[target_indices] - pos, axis=1)
        relevant = current_distance <= self.cfg.tracking_radius * 1.25
        if not np.any(relevant):
            return 0.0
        indices = target_indices[relevant]
        horizon = max(1, min(int(self.cfg.avoidance_prediction_horizon), 3))
        dt = float(self.current_decision_dt) * horizon
        predicted_target = self.dynamic_targets[indices] + self.target_velocity[indices] * dt
        predicted_uav = pos + self.uav_step_vector(int(self.headings[i])).astype(float) * horizon
        future_distance = np.linalg.norm(predicted_target - predicted_uav, axis=1)
        margin = self.cfg.tracking_radius - future_distance
        risk = np.clip(1.0 - margin / max(self.cfg.tracking_radius * 0.5, 1e-6), 0.0, 1.0)
        return float(np.max(risk))

    def _empty_heterogeneous_graph(self) -> dict:
        c = self.cfg
        track_count = c.n_targets if c.hetero_track_nodes <= 0 else c.hetero_track_nodes
        return {
            "track_nodes": np.zeros((c.n_uavs, track_count, 13), dtype=np.float32),
            "track_mask": np.zeros((c.n_uavs, track_count), dtype=bool),
            "frontier_nodes": np.zeros((c.n_uavs, c.hetero_frontier_nodes, 8), dtype=np.float32),
            "frontier_mask": np.zeros((c.n_uavs, c.hetero_frontier_nodes), dtype=bool),
            # Legacy combined slots are retained for old checkpoints/tools, but
            # the current HGAT consumes the split tensors above.
            "target_nodes": np.zeros((c.n_uavs, c.hetero_target_nodes, 13), dtype=np.float32),
            "target_mask": np.zeros((c.n_uavs, c.hetero_target_nodes), dtype=bool),
            "obstacle_nodes": np.zeros((c.n_uavs, c.hetero_obstacle_nodes, 8), dtype=np.float32),
            "obstacle_mask": np.zeros((c.n_uavs, c.hetero_obstacle_nodes), dtype=bool),
        }

    def heterogeneous_graph_observation(self) -> dict:
        """Build local entity nodes without exposing unobserved target truth.

        Target slots contain currently visible physical targets followed by TPM
        frontier belief nodes. Obstacle slots contain only locally observable
        obstacles. This keeps decentralized execution information-causal.
        """
        c = self.cfg
        graph = self._empty_heterogeneous_graph()
        for i in range(c.n_uavs):
            if self.disabled_uavs[i]:
                continue
            pos = self.positions[i].astype(float)
            target_slot = 0
            causal_indices = self.causal_target_indices(i)
            if causal_indices.size:
                indices = causal_indices
                rel = self.dynamic_targets[indices] - pos
                dist = np.linalg.norm(rel, axis=1)
                within_entity_range = dist <= c.entity_observation_radius
                visible_order = np.argsort(dist[within_entity_range]) if np.any(within_entity_range) else []
                visible_indices = indices[within_entity_range]
                for order_idx in visible_order:
                    target_idx = int(visible_indices[int(order_idx)])
                    delta = self.dynamic_targets[target_idx] - pos
                    distance = float(np.linalg.norm(delta))
                    if target_slot >= graph["track_nodes"].shape[1]:
                        break
                    graph["track_nodes"][i, target_slot] = np.array(
                        [
                            delta[0] / c.grid_size,
                            delta[1] / c.grid_size,
                            self.target_velocity[target_idx, 0] / max(c.uav_speed, 1e-6),
                            self.target_velocity[target_idx, 1] / max(c.uav_speed, 1e-6),
                            min(distance / c.grid_size, 1.0),
                            1.0,
                            0.0,
                            float(self.local_revisit_map[i, int(np.clip(self.dynamic_targets[target_idx, 1], 0, c.grid_size - 1)), int(np.clip(self.dynamic_targets[target_idx, 0], 0, c.grid_size - 1))]),
                            float(distance <= c.sensor_radius),
                            float(self.tracked_targets[target_idx]),
                            0.0,
                            0.0,
                            self.local_target_loss_risk(i, np.asarray([target_idx], dtype=int)),
                        ],
                        dtype=np.float32,
                    )
                    graph["track_mask"][i, target_slot] = True
                    target_slot += 1
            local_uncertainty = np.exp(-np.abs(self.local_q[i]))
            local_prob = log_odds_to_prob(self.local_q[i])
            frontier_slot = 0
            for frontier, prob_value, uncertainty_value, revisit_value in self._frontier_candidates(
                i, local_prob, local_uncertainty, c.hetero_frontier_nodes
            ):
                delta = frontier - pos
                graph["frontier_nodes"][i, frontier_slot] = np.array(
                    [
                        delta[0] / c.grid_size,
                        delta[1] / c.grid_size,
                        min(float(np.linalg.norm(delta)) / c.grid_size, 1.0),
                        prob_value,
                        uncertainty_value,
                        revisit_value,
                        float(revisit_value >= max(prob_value, uncertainty_value)),
                        min(float(self.belief_age[i]) / 10.0, 1.0)
                        if c.freshness_fusion_enabled
                        else 0.0,
                    ],
                    dtype=np.float32,
                )
                graph["frontier_mask"][i, frontier_slot] = True
                frontier_slot += 1

            if len(self.obstacles):
                rel = self.obstacles - pos
                dist = np.linalg.norm(rel, axis=1)
                visible = np.flatnonzero(dist <= c.entity_observation_radius)
                visible = visible[np.argsort(dist[visible])][: c.hetero_obstacle_nodes]
                for slot, obstacle_idx in enumerate(visible):
                    delta = rel[obstacle_idx]
                    velocity = self.obstacle_velocity[obstacle_idx]
                    relative_speed_sq = float(np.dot(velocity, velocity))
                    ttc = max(0.0, -float(np.dot(delta, velocity)) / max(relative_speed_sq, 1e-6))
                    clearance = float(dist[obstacle_idx] - c.obstacle_radius - c.safe_radius)
                    graph["obstacle_nodes"][i, slot] = np.array(
                        [
                            delta[0] / c.grid_size,
                            delta[1] / c.grid_size,
                            velocity[0] / max(c.uav_speed, 1e-6),
                            velocity[1] / max(c.uav_speed, 1e-6),
                            min(float(dist[obstacle_idx]) / c.grid_size, 1.0),
                            np.clip(clearance / c.grid_size, -1.0, 1.0),
                            min(ttc / max(c.avoidance_prediction_horizon, 1), 1.0),
                            1.0,
                        ],
                        dtype=np.float32,
                    )
                    graph["obstacle_mask"][i, slot] = True
        # Populate the deprecated combined representation for consumers that
        # have not migrated yet. Track estimates have priority over frontiers.
        for i in range(c.n_uavs):
            legacy_slot = 0
            for slot in np.flatnonzero(graph["track_mask"][i]):
                if legacy_slot >= c.hetero_target_nodes:
                    break
                graph["target_nodes"][i, legacy_slot] = graph["track_nodes"][i, slot]
                graph["target_mask"][i, legacy_slot] = True
                legacy_slot += 1
            for slot in np.flatnonzero(graph["frontier_mask"][i]):
                if legacy_slot >= c.hetero_target_nodes:
                    break
                frontier = graph["frontier_nodes"][i, slot]
                graph["target_nodes"][i, legacy_slot, [0, 1, 4, 5, 6, 7, 11, 12]] = frontier
                graph["target_nodes"][i, legacy_slot, 10] = 1.0
                graph["target_mask"][i, legacy_slot] = True
                legacy_slot += 1
        return graph

    def _frontier_candidates(
        self, i: int, target_prob: np.ndarray, uncertainty: np.ndarray, count: int
    ) -> list[tuple[np.ndarray, float, float, float]]:
        if count <= 0:
            return []
        c = self.cfg
        active = np.flatnonzero(~self.disabled_uavs)
        rank = int(np.flatnonzero(active == i)[0]) if i in active else 0
        x_lo = int(rank * c.grid_size / max(len(active), 1))
        x_hi = max(x_lo + 1, int((rank + 1) * c.grid_size / max(len(active), 1)))
        prob_sector = target_prob[:, x_lo:x_hi]
        uncertainty_sector = uncertainty[:, x_lo:x_hi]
        revisit_sector = self.local_revisit_map[i, :, x_lo:x_hi]
        revisit_weight = 0.65 if c.revisit_map_enabled else 0.0
        score = 0.15 * prob_sector + uncertainty_sector + revisit_weight * revisit_sector
        working = score.copy()
        result = []
        for _ in range(min(count, working.size)):
            fy, fx = np.unravel_index(int(np.argmax(working)), working.shape)
            value = float(working[fy, fx])
            if value < 0:
                break
            result.append(
                (
                    np.array([fx + x_lo, fy], dtype=float),
                    float(np.clip(prob_sector[fy, fx], 0.0, 1.0)),
                    float(np.clip(uncertainty_sector[fy, fx], 0.0, 1.0)),
                    float(np.clip(revisit_sector[fy, fx], 0.0, 1.0)),
                )
            )
            y1, y2 = max(0, fy - 2), min(working.shape[0], fy + 3)
            x1, x2 = max(0, fx - 2), min(working.shape[1], fx + 3)
            working[y1:y2, x1:x2] = -1.0
        return result

    def state_vector(self) -> np.ndarray:
        if not hasattr(self, "obstacles"):
            return np.zeros(self.state_dim())
        p = log_odds_to_prob(self.q_map)
        stats = np.array(
            [
                self.coverage(),
                float(np.exp(-np.abs(self.q_map)).mean()),
                float(p.max()),
                float(np.mean(getattr(self, "disconnected_uavs", np.zeros(self.cfg.n_uavs)))),
                float((self.last_graph > 0).mean()),
            ]
        )
        return np.concatenate([self.agent_observation_vectors().ravel(), stats])

    def comm_adjacency(self) -> np.ndarray:
        graph = self.last_graph if hasattr(self, "last_graph") else np.zeros((self.cfg.n_uavs, self.cfg.n_uavs), dtype=float)
        adjacency = graph > 0
        np.fill_diagonal(adjacency, True)
        return adjacency.astype(float)

    def obs_dim(self) -> int:
        return 41

    def state_dim(self) -> int:
        return self.cfg.n_uavs * self.obs_dim() + 5

    def search_benefit_matrix(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized J1..J5 benefits for every nominal action.

        Columns are target probability, environmental uncertainty, revisit
        urgency, communication retention and locally observable safety. This
        function is information-causal: it uses UAV i's local maps, current
        communication neighbors and locally observable obstacles only.
        """
        c = self.cfg
        valid = self.nominal_action_mask(i)
        benefits = np.zeros((9, 5), dtype=np.float32)
        local_p = log_odds_to_prob(self.local_q[i])
        local_e = np.exp(-np.abs(self.local_q[i]))
        local_s = self.local_revisit_map[i]
        causal_targets = self.causal_target_indices(i)
        predicted_targets = np.empty((0, 2), dtype=float)
        if len(causal_targets):
            predicted_targets = (
                self.dynamic_targets[causal_targets]
                + self.target_velocity[causal_targets] * self.current_decision_dt
            )
        prediction_weight = float(np.clip(c.target_prediction_weight, 0.0, 1.0))
        peer_indices = np.flatnonzero(self.last_graph[i] > 0) if hasattr(self, "last_graph") else np.empty(0, dtype=int)
        obstacle_indices = np.empty(0, dtype=int)
        if len(self.obstacles):
            obstacle_dist = np.linalg.norm(self.obstacles - self.positions[i], axis=1)
            obstacle_indices = np.flatnonzero(obstacle_dist <= c.entity_observation_radius)
        for action in np.flatnonzero(valid):
            pos = self.positions[i].astype(float) if action == 8 else self.positions[i].astype(float) + self.uav_step_vector(int(action))
            x0, y0 = np.clip(pos.astype(int), 0, c.grid_size - 1)
            y1, y2 = max(0, y0 - 1), min(c.grid_size, y0 + 2)
            x1, x2 = max(0, x0 - 1), min(c.grid_size, x0 + 2)
            map_target_value = float(np.mean(local_p[y1:y2, x1:x2]))
            if len(predicted_targets):
                target_distance = np.linalg.norm(predicted_targets - pos, axis=1)
                desired_distance = 0.75 * c.tracking_radius
                tracking_error = np.min(np.abs(target_distance - desired_distance))
                predicted_tracking_value = float(
                    np.exp(-tracking_error / max(c.tracking_radius, 1e-6))
                )
                benefits[action, 0] = (
                    (1.0 - prediction_weight) * map_target_value
                    + prediction_weight * predicted_tracking_value
                )
            else:
                benefits[action, 0] = map_target_value
            benefits[action, 1] = float(np.mean(local_e[y1:y2, x1:x2]))
            benefits[action, 2] = float(np.mean(local_s[y1:y2, x1:x2]))
            if len(peer_indices):
                peer_dist = np.linalg.norm(self.positions[peer_indices].astype(float) - pos, axis=1)
                benefits[action, 3] = float(np.mean(np.exp(-peer_dist / max(c.comm_radius, 1e-6))))
            else:
                benefits[action, 3] = 0.5
            clearances = []
            if len(obstacle_indices):
                predicted_obstacles = self.obstacles[obstacle_indices] + self.obstacle_velocity[obstacle_indices] * self.current_decision_dt
                clearances.extend(
                    (np.linalg.norm(predicted_obstacles - pos, axis=1) - c.obstacle_radius).tolist()
                )
            if len(peer_indices):
                clearances.extend((np.linalg.norm(self.positions[peer_indices] - pos, axis=1) - c.uav_collision_radius).tolist())
            benefits[action, 4] = float(min(clearances)) if clearances else c.entity_observation_radius
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx):
            for column in range(5):
                values = benefits[valid_idx, column]
                lo, hi = float(values.min()), float(values.max())
                benefits[valid_idx, column] = (values - lo) / (hi - lo) if hi - lo > 1e-8 else 0.5
        if not c.revisit_map_enabled:
            benefits[:, 2] = 0.0
        return benefits, valid

    def actions_from_search_weights(self, weights: np.ndarray) -> np.ndarray:
        """Map per-UAV adaptive J1..J5 weights to executable nominal actions."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.cfg.n_uavs, 5):
            raise ValueError(f"Expected search weights shape {(self.cfg.n_uavs, 5)}, got {weights.shape}.")
        actions = np.full(self.cfg.n_uavs, 8, dtype=int)
        for i in range(self.cfg.n_uavs):
            if self.disabled_uavs[i]:
                continue
            benefits, valid = self.search_benefit_matrix(i)
            normalized_weights = np.clip(weights[i], 1e-6, None)
            normalized_weights /= normalized_weights.sum()
            scores = benefits @ normalized_weights
            for action in np.flatnonzero(valid):
                if action == 8:
                    scores[action] -= 0.08
                else:
                    turn = min((action - self.headings[i]) % 8, (self.headings[i] - action) % 8)
                    scores[action] += 0.03 / (1.0 + turn)
            scores[~valid] = -np.inf
            actions[i] = int(np.argmax(scores))
        return actions

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def _local_patch_value(self, field: np.ndarray, pos: np.ndarray) -> float:
        x0, y0 = pos.astype(int)
        y1, y2 = max(0, y0 - 1), min(self.cfg.grid_size, y0 + 2)
        x1, x2 = max(0, x0 - 1), min(self.cfg.grid_size, x0 + 2)
        return float(field[y1:y2, x1:x2].mean())

    def _nearest_obstacle(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        if len(self.obstacles) == 0:
            return np.zeros(2), float(self.cfg.grid_size)
        rel = self.obstacles - pos
        dist = np.linalg.norm(rel, axis=1)
        idx = int(np.argmin(dist))
        return rel[idx], float(dist[idx])

    def _nearest_target_feature(self, pos: np.ndarray, hotspot: np.ndarray) -> tuple[np.ndarray, float, int]:
        if len(self.dynamic_targets) == 0:
            rel = hotspot.astype(float) - pos
            return rel / self.cfg.grid_size, float(np.linalg.norm(rel)), 0
        active_targets = self.dynamic_targets[~self.completed_targets]
        if len(active_targets) == 0:
            rel = hotspot.astype(float) - pos
            return rel / self.cfg.grid_size, float(np.linalg.norm(rel)), 0
        rel_all = active_targets - pos
        dist = np.linalg.norm(rel_all, axis=1)
        visible = int(np.count_nonzero(dist <= self.cfg.sensor_radius))
        if visible:
            idx = int(np.argmin(dist))
            rel = rel_all[idx]
            return rel / self.cfg.grid_size, float(dist[idx]), visible
        rel = hotspot.astype(float) - pos
        return rel / self.cfg.grid_size, float(np.linalg.norm(rel)), 0

    def _collides_static(self, pos: np.ndarray) -> bool:
        if len(self.obstacles) == 0:
            return False
        planning_clearance = self.cfg.obstacle_radius + 0.75 * self.cfg.safe_radius
        return bool(np.min(np.linalg.norm(self.obstacles - pos, axis=1)) <= planning_clearance)

    def _collides_predicted_obstacle(self, pos: np.ndarray) -> bool:
        if len(self.obstacles) == 0:
            return False
        horizon = max(1, int(self.cfg.avoidance_prediction_horizon))
        dt = getattr(self, "current_decision_dt", self.cfg.decision_dt)
        planning_clearance = self.cfg.obstacle_radius + self.cfg.safe_radius + self.cfg.obstacle_prediction_buffer
        for step in range(1, horizon + 1):
            predicted = self.obstacles + self.obstacle_velocity * dt * step
            predicted = np.clip(predicted, 0.5, self.cfg.grid_size - 1.5)
            if np.min(np.linalg.norm(predicted - pos, axis=1)) <= planning_clearance:
                return True
        return False

    def _predicted_clearance(self, pos: np.ndarray, agent_idx: int) -> float:
        clearance = float(self.cfg.grid_size)
        if len(self.obstacles):
            dt = getattr(self, "current_decision_dt", self.cfg.decision_dt)
            horizon = max(1, int(self.cfg.avoidance_prediction_horizon))
            for step in range(0, horizon + 1):
                predicted = self.obstacles + self.obstacle_velocity * dt * step
                predicted = np.clip(predicted, 0.5, self.cfg.grid_size - 1.5)
                clearance = min(clearance, float(np.min(np.linalg.norm(predicted - pos, axis=1))))
        for j in range(self.cfg.n_uavs):
            if j != agent_idx and not self.disabled_uavs[j]:
                clearance = min(clearance, float(np.linalg.norm(self.positions[j] - pos)))
        return clearance

    def _too_close_to_agent(self, i: int, pos: np.ndarray) -> bool:
        for j in range(self.cfg.n_uavs):
            if (
                i != j
                and not self.disabled_uavs[j]
                and np.linalg.norm(self.positions[j] - pos) < self.cfg.safe_radius + self.cfg.pair_prediction_buffer
            ):
                return True
        return False

    def _count_pair_conflicts(self) -> int:
        conflicts = 0
        for i in range(self.cfg.n_uavs):
            if self.disabled_uavs[i]:
                continue
            for j in range(i):
                if self.disabled_uavs[j]:
                    continue
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.cfg.safe_radius:
                    conflicts += 1
        return conflicts

    def _count_uav_collisions(self) -> tuple[int, set[int]]:
        collisions = 0
        crashed: set[int] = set()
        for i in range(self.cfg.n_uavs):
            if self.disabled_uavs[i]:
                continue
            for j in range(i):
                if self.disabled_uavs[j]:
                    continue
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.cfg.uav_collision_radius:
                    collisions += 1
                    crashed.update((i, j))
        return collisions, crashed

    def _count_obstacle_conflicts(self) -> tuple[int, set[int]]:
        if len(self.obstacles) == 0:
            return 0, set()
        conflicts = 0
        crashed: set[int] = set()
        for i, pos in enumerate(self.positions):
            if self.disabled_uavs[i]:
                continue
            if np.min(np.linalg.norm(self.obstacles - pos, axis=1)) <= self.cfg.obstacle_radius:
                conflicts += 1
                crashed.add(i)
        return conflicts, crashed

    def _apply_crashes(self, crash_ids: set[int]) -> set[int]:
        newly_crashed = {int(i) for i in crash_ids if not self.crashed_uavs[int(i)]}
        if not newly_crashed:
            return set()
        self._broadcast_crashed_uav_maps(newly_crashed)
        for i in newly_crashed:
            self.crashed_uavs[i] = True
            self.disabled_uavs[i] = True
            self.disconnected_uavs[i] = True
        return newly_crashed

    def _broadcast_crashed_uav_maps(self, crashed_ids: set[int]) -> None:
        receivers = ~self.disabled_uavs & ~self.disconnected_uavs
        for crashed in crashed_ids:
            d = np.linalg.norm(self.positions - self.positions[crashed], axis=1)
            nearby = np.flatnonzero(receivers & (d <= self.cfg.comm_radius))
            for j in nearby:
                candidates = np.stack([self.local_q[j], self.local_q[crashed]])
                ages = np.stack(
                    [
                        self.local_belief_age_map[j],
                        np.minimum(
                            self.local_belief_age_map[crashed] + 1.0,
                            self.cfg.max_belief_age,
                        ),
                    ]
                )
                self.local_q[j], self.local_belief_age_map[j] = (
                    self._fuse_belief_sources(
                        candidates, ages, np.array([1.0, 1.0], dtype=float)
                    )
                )
                self.local_revisit_map[j] = np.minimum(
                    self.local_revisit_map[j], self.local_revisit_map[crashed]
                )
        candidates = np.vstack([self.q_map[None, :, :], self.local_q])
        best = np.argmax(np.abs(candidates), axis=0)
        self.q_map = np.take_along_axis(candidates, best[None, :, :], axis=0)[0]


def nominal_search_policy(env: WeakCommBeliefGraphEnv) -> np.ndarray:
    """Strictly local search/tracking policy with no obstacle or pair terms."""
    c = env.cfg
    active_order = [i for i in range(c.n_uavs) if not env.disabled_uavs[i]]
    active_rank = {agent_idx: rank for rank, agent_idx in enumerate(active_order)}
    active_count = max(len(active_order), 1)
    actions = []
    for i in range(c.n_uavs):
        if env.disabled_uavs[i]:
            actions.append(8)
            continue
        p = log_odds_to_prob(env.local_q[i])
        uncertainty = np.exp(-np.abs(env.local_q[i]))
        undecided = ((p > c.theta0) & (p < c.theta1)).astype(float)
        frontier_score = uncertainty * (0.4 + undecided) + 0.08 * p
        active_target_indices = env.causal_target_indices(i)
        tracked_targets = env.dynamic_targets[active_target_indices] if len(active_target_indices) else np.empty((0, 2))
        local_tracking_phase = bool(len(active_target_indices))
        mask = env.nominal_action_mask(i)
        rank = active_rank.get(i, 0)
        x_lo = int(rank * c.grid_size / active_count)
        x_hi = max(x_lo + 1, int((rank + 1) * c.grid_size / active_count))
        sector = frontier_score[:, x_lo:x_hi]
        if sector.size:
            fy, fx = np.unravel_index(int(np.argmax(sector)), sector.shape)
            frontier = np.array([fx + x_lo, fy], dtype=float)
        else:
            frontier = np.array(np.unravel_index(int(np.argmax(frontier_score)), frontier_score.shape))[::-1]
        assigned_target = None
        assigned_streak = 0
        if len(tracked_targets):
            distances = np.linalg.norm(tracked_targets - env.positions[i].astype(float), axis=1)
            target_slot = int(np.argsort(distances)[i % len(distances)])
            assigned_target = tracked_targets[target_slot]
            # Do not use the team-global tracking streak in a decentralized
            # heuristic. Current local/communicated visibility drives urgency.
            assigned_streak = 0
        candidates, scores = [], []
        for action in np.flatnonzero(mask):
            pos = env.positions[i] if action == 8 else env.positions[i] + env.uav_step_vector(int(action))
            x0, y0 = pos.astype(int)
            y1, y2 = max(0, y0 - 1), min(c.grid_size, y0 + 2)
            x1, x2 = max(0, x0 - 1), min(c.grid_size, x0 + 2)
            heading_score = float(uncertainty[y1:y2, x1:x2].sum() + 0.1 * p[y1:y2, x1:x2].sum())
            frontier_pull = 1.0 / (1.0 + np.linalg.norm(pos - frontier))
            target_pull = 0.0
            if len(tracked_targets):
                nearest = float(np.min(np.linalg.norm(tracked_targets - pos, axis=1)))
                target_pull = 1.0 / (1.0 + abs(nearest - c.tracking_radius))
                if assigned_target is not None:
                    assigned_distance = float(np.linalg.norm(assigned_target - pos))
                    urgency = 1.0 + assigned_streak / max(c.target_completion_steps, 1)
                    target_pull += urgency / (1.0 + abs(assigned_distance - 0.75 * c.tracking_radius))
            velocity_score = 0.25 if action == 8 else 1.0
            turn = 0 if action == 8 else min((action - env.headings[i]) % 8, (env.headings[i] - action) % 8)
            smooth_score = 1.0 / (1.0 + turn)
            score = (
                (0.45 if local_tracking_phase else 1.0) * heading_score
                + (0.9 if local_tracking_phase else 2.2) * frontier_pull
                + (7.0 if local_tracking_phase else 2.5) * target_pull
                + 0.1 * velocity_score
                + 0.2 * smooth_score
            )
            candidates.append(int(action))
            scores.append(float(score))
        actions.append(candidates[int(np.argmax(scores))] if candidates else 8)
    return np.asarray(actions, dtype=int)


def dwa_orca_search_policy(env: WeakCommBeliefGraphEnv) -> np.ndarray:
    """Strict-local DWA-style scoring over Wang's discrete action set."""

    c = env.cfg
    actions = []
    planned = []
    active_order = [i for i in range(c.n_uavs) if not env.disabled_uavs[i]]
    active_rank = {agent_idx: rank for rank, agent_idx in enumerate(active_order)}
    active_count = max(len(active_order), 1)
    for i in range(c.n_uavs):
        if env.disabled_uavs[i]:
            actions.append(8)
            continue
        p = log_odds_to_prob(env.local_q[i])
        uncertainty = np.exp(-np.abs(env.local_q[i]))
        undecided = ((p > c.theta0) & (p < c.theta1)).astype(float)
        frontier_score = uncertainty * (0.4 + undecided) + 0.08 * p
        active_target_indices = env.causal_target_indices(i)
        tracked_targets = env.dynamic_targets[active_target_indices] if len(active_target_indices) else np.empty((0, 2))
        local_tracking_phase = bool(len(active_target_indices))
        mask = env.action_mask(i)
        rank = active_rank.get(i, 0)
        x_lo = int(rank * c.grid_size / active_count)
        x_hi = int((rank + 1) * c.grid_size / active_count)
        sector = frontier_score[:, x_lo:x_hi]
        if sector.size and sector.max() > 0:
            fy, fx = np.unravel_index(np.argmax(sector), sector.shape)
            frontier = np.array([fx + x_lo, fy])
        else:
            frontier = np.array(np.unravel_index(np.argmax(frontier_score), frontier_score.shape))[::-1]
        assigned_target = None
        assigned_streak = 0
        if len(tracked_targets):
            dists = np.linalg.norm(tracked_targets - env.positions[i].astype(float), axis=1)
            order = np.argsort(dists)
            target_slot = int(order[i % len(order)])
            assigned_target = tracked_targets[target_slot]
            assigned_streak = 0
        scores = []
        candidates = []
        for a in np.flatnonzero(mask):
            pos = env.positions[i] if a == 8 else env.positions[i] + env.uav_step_vector(int(a))
            x0, y0 = pos.astype(int)
            y1, y2 = max(0, y0 - 1), min(c.grid_size, y0 + 2)
            x1, x2 = max(0, x0 - 1), min(c.grid_size, x0 + 2)
            heading_score = float(uncertainty[y1:y2, x1:x2].sum() + 0.1 * p[y1:y2, x1:x2].sum())
            frontier_pull = 1.0 / (1.0 + np.linalg.norm(pos - frontier))
            if len(tracked_targets):
                target_distances = np.linalg.norm(tracked_targets - pos, axis=1)
                nearest_target_dist = float(np.min(target_distances))
                target_pull = 1.0 / (1.0 + abs(nearest_target_dist - c.tracking_radius))
                if assigned_target is not None:
                    assigned_dist = float(np.linalg.norm(assigned_target - pos))
                    completion_urgency = 1.0 + assigned_streak / max(c.target_completion_steps, 1)
                    target_pull += completion_urgency / (1.0 + abs(assigned_dist - 0.75 * c.tracking_radius))
            else:
                target_pull = 0.0
            _, obs_dist = env._nearest_obstacle(pos.astype(float))
            predicted_clearance = env._predicted_clearance(pos.astype(float), i)
            obstacle_score = min(predicted_clearance / max(c.safe_radius * 3.5, 1e-6), 1.0)
            velocity_score = 0.25 if a == 8 else 1.0
            turn = 0 if a == 8 else min((a - env.headings[i]) % 8, (env.headings[i] - a) % 8)
            smooth_score = 1.0 / (1.0 + turn)
            pair_score = 1.0
            pair_risk = 0.0
            for other in list(planned) + [env.positions[j] for j in active_order if j != i]:
                d = np.linalg.norm(pos - other)
                guard = c.safe_radius * 2.25
                if d < guard:
                    pair_risk += (guard - d) / guard
            pair_score -= pair_risk
            risk_count = int(np.sum(np.linalg.norm(env.obstacles - pos, axis=1) < c.safe_radius * 3.0)) if len(env.obstacles) else 0
            obstacle_risk = 1.0 if env._collides_predicted_obstacle(pos.astype(float)) else 0.0
            k_avoid = (risk_count + 1.0) / (1.0 + max(predicted_clearance, 1e-6))
            score = (
                (0.45 if local_tracking_phase else 1.0) * heading_score
                + (0.9 if local_tracking_phase else 2.2) * frontier_pull
                + (7.0 if local_tracking_phase else 2.5) * target_pull
                + (0.75 + k_avoid) * obstacle_score
                + 0.1 * velocity_score
                + 0.2 * smooth_score
                + pair_score
                - 1.8 * obstacle_risk
                - 1.2 * pair_risk
            )
            scores.append(score)
            candidates.append(int(a))
        chosen = candidates[int(np.argmax(scores))] if candidates else 8
        next_pos = env.positions[i] if chosen == 8 else env.positions[i] + env.uav_step_vector(chosen)
        planned.append(next_pos)
        actions.append(chosen)
    return np.asarray(actions, dtype=int)


def run_weak_comm_demo(seed: int = 17) -> dict:
    """Run the integrated heuristic policy in the fused weak-communication environment."""
    env = WeakCommBeliefGraphEnv(WeakCommConfig(seed=seed))
    history = []
    for _ in range(env.cfg.search_steps):
        actions = dwa_orca_search_policy(env)
        result = env.step_joint(actions)
        history.append(
            {
                "step": env.t,
                "coverage": result["coverage"],
                "reward": result["reward"],
                "collisions": result["collisions"],
                "graph_density": result["graph_density"],
                "target_discovery_rate": result["target_discovery_rate"],
                "target_tracking_rate": result["target_tracking_rate"],
                "task_phase": result["task_phase"],
                "lost_tracks": result["lost_tracks"],
                "escort_score": result["escort_score"],
                "positions": env.positions.copy().tolist(),
                "targets": env.dynamic_targets.copy().tolist(),
                "obstacles": env.obstacles.copy().tolist(),
                "actions": actions.tolist(),
            }
        )
        if result["done"]:
            break
    return {
        "paper_stack": "Wang TPM + Zhao graph attention + Chang DWA-ORCA",
        "experiment_kind": "integrated_heuristic_core_mechanism_demo",
        "steps": env.t,
        "coverage": env.coverage(),
        "targets": env.targets.tolist(),
        "final_dynamic_targets": env.dynamic_targets.tolist(),
        "obstacles": env.obstacles.tolist(),
        "history": history,
    }
