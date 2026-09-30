"""Core-mechanism scaffold inspired by Wang et al. 2025.

This module implements target probability map (TPM) updates and masked
8-neighbor UAV motion. Weak communication and outage effects are modeled in
the integrated environment without explicit channel-selection actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


DIRECTIONS = np.array(
    [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
    ],
    dtype=int,
)


@dataclass
class WangConfig:
    n_uavs: int = 3
    n_targets: int = 15
    grid_size: int = 50
    uav_speed: float = 1.0
    decision_dt: float = 1.0
    max_turn_steps: int = 1
    allow_hover: bool = False
    normalize_diagonal_speed: bool = True
    n_jammers: int = 3
    search_steps: int = 300
    pd: float = 0.9
    pf: float = 0.1
    theta0: float = 0.05
    theta1: float = 0.95
    tpm_prior_base: float = 0.5
    tpm_target_prior_enabled: bool = False
    tpm_target_prior_strength: float = 0.72
    tpm_target_prior_sigma: float = 0.75
    kq: float = 1.0
    base_latency_ms: float = 20.0
    timeout_latency_ms: float = 200.0
    max_throughput_kbps: float = 1000.0
    seed: int = 7


def prob_to_log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(1.0 / p - 1.0)


def log_odds_to_prob(q: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(q))


class WangSearchInterferenceEnv:
    """Grid search environment with optional communication-success masking."""

    def __init__(self, cfg: WangConfig | None = None):
        self.cfg = cfg or WangConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    def reset(self) -> dict:
        c = self.cfg
        self.t = 0
        self.spectrum_t = 0
        self.positions = np.array([[0, 0], [0, c.grid_size - 1], [c.grid_size - 1, 0]], dtype=float)[
            : c.n_uavs
        ]
        if c.n_uavs > 3:
            extra = self.rng.integers(0, c.grid_size, size=(c.n_uavs - 3, 2))
            self.positions = np.vstack([self.positions, extra])
        self.targets = self._sample_targets()
        self.target_grid = np.zeros((c.grid_size, c.grid_size), dtype=bool)
        self.target_grid[self.targets[:, 1], self.targets[:, 0]] = True
        self.q_map = prob_to_log_odds(self._initial_tpm())
        self.local_q = np.repeat(self.q_map[None, :, :], c.n_uavs, axis=0)
        self.headings = self._initial_headings_from_prior()
        self.jammer_phase = self.rng.integers(0, max(c.grid_size, 1), size=c.n_jammers)
        return self.observe_search()

    def uav_step_vector(self, action: int) -> np.ndarray:
        direction = DIRECTIONS[action].astype(float)
        if self.cfg.normalize_diagonal_speed:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
        decision_dt = getattr(self, "current_decision_dt", self.cfg.decision_dt)
        return direction * self.cfg.uav_speed * decision_dt

    def _sample_targets(self) -> np.ndarray:
        c = self.cfg
        all_cells = np.array([(x, y) for y in range(c.grid_size) for x in range(c.grid_size)])
        idx = self.rng.choice(len(all_cells), size=c.n_targets, replace=False)
        return all_cells[idx]

    def _initial_tpm(self) -> np.ndarray:
        c = self.cfg
        p = np.full((c.grid_size, c.grid_size), c.tpm_prior_base, dtype=float)
        if not c.tpm_target_prior_enabled or c.tpm_target_prior_strength <= 0.0:
            return np.clip(p, c.theta0, c.theta1)
        xs, ys = np.meshgrid(np.arange(c.grid_size), np.arange(c.grid_size))
        sigma = c.tpm_target_prior_sigma
        for x0, y0 in self.targets:
            bump = np.exp(-0.5 * (((xs - x0) / sigma) ** 2 + ((ys - y0) / sigma) ** 2))
            p = 1.0 - (1.0 - p) * (1.0 - c.tpm_target_prior_strength * bump)
        return np.clip(p, c.theta0, c.theta1)

    def _initial_headings_from_prior(self) -> np.ndarray:
        c = self.cfg
        prior = log_odds_to_prob(self.q_map)
        if float(prior.max() - prior.min()) <= 1e-9:
            goal = np.array([(c.grid_size - 1) / 2.0, (c.grid_size - 1) / 2.0], dtype=float)
        else:
            y, x = np.unravel_index(int(np.argmax(prior)), prior.shape)
            goal = np.array([x, y], dtype=float)
        directions = DIRECTIONS.astype(float)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        headings = []
        for pos in self.positions:
            vec = goal - pos
            norm = np.linalg.norm(vec)
            headings.append(0 if norm <= 1e-9 else int(np.argmax(directions @ (vec / norm))))
        return np.asarray(headings, dtype=int)

    def observe_search(self) -> dict:
        return {
            "local_q": self.local_q.copy(),
            "positions": self.positions.copy(),
            "headings": self.headings.copy(),
            "coverage": self.coverage(),
            "action_masks": np.stack([self.action_mask(i) for i in range(self.cfg.n_uavs)]),
        }

    def action_mask(self, i: int) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        max_turn = max(0, min(4, int(self.cfg.max_turn_steps)))
        allowed = {(self.headings[i] + delta) % 8 for delta in range(-max_turn, max_turn + 1)}
        for a in allowed:
            pos = self.positions[i] + self.uav_step_vector(a)
            if np.all((0 <= pos) & (pos < self.cfg.grid_size)):
                mask[a] = True
        mask[8] = bool(self.cfg.allow_hover)
        if not mask[:8].any():
            mask[8] = True
        return mask

    def step_search(self, actions: Iterable[int], communication_success: np.ndarray | None = None) -> dict:
        c = self.cfg
        prev_uncertainty = self.uncertainty()
        actions = np.asarray(list(actions), dtype=int)
        for i, a in enumerate(actions):
            if a == 8:
                continue
            if self.action_mask(i)[a]:
                self.positions[i] += self.uav_step_vector(a)
                self.headings[i] = a
        for i in range(c.n_uavs):
            self._scan_and_update(i)
        if communication_success is None:
            communication_success = np.ones(c.n_uavs, dtype=bool)
        transmitted = self.local_q[communication_success]
        if len(transmitted):
            candidates = np.vstack([self.q_map[None, :, :], transmitted])
            best = np.argmax(np.abs(candidates), axis=0)
            self.q_map = np.take_along_axis(candidates, best[None, :, :], axis=0)[0]
        self.local_q[:] = self.q_map
        self.t += 1
        reward = prev_uncertainty - self.uncertainty()
        return {
            "obs": self.observe_search(),
            "reward": float(reward),
            "coverage": self.coverage(),
            "done": self.coverage() >= 0.999 or self.t >= c.search_steps,
        }

    def _scan_and_update(self, i: int) -> None:
        c = self.cfg
        x0, y0 = self.positions[i].astype(int)
        hit_delta = math.log(c.pf / c.pd)
        miss_delta = math.log((1.0 - c.pf) / (1.0 - c.pd))
        for y in range(max(0, y0 - 1), min(c.grid_size, y0 + 2)):
            for x in range(max(0, x0 - 1), min(c.grid_size, x0 + 2)):
                target = self.target_grid[y, x]
                detected = self.rng.random() < (c.pd if target else c.pf)
                self.local_q[i, y, x] += hit_delta if detected else miss_delta

    def uncertainty(self) -> float:
        return float(np.exp(-self.cfg.kq * np.abs(self.q_map)).sum())

    def coverage(self) -> float:
        p = log_odds_to_prob(self.q_map)
        decided = (p <= self.cfg.theta0) | (p >= self.cfg.theta1)
        return float(decided.mean())


def uncertainty_search_policy(env: WangSearchInterferenceEnv) -> np.ndarray:
    actions = []
    p = log_odds_to_prob(env.q_map)
    uncertainty = np.exp(-np.abs(env.q_map))
    occupied_next = set()
    for i in range(env.cfg.n_uavs):
        mask = env.action_mask(i)
        best_score = -1e9
        best_action = 8
        for a in np.flatnonzero(mask):
            pos = env.positions[i] if a == 8 else env.positions[i] + env.uav_step_vector(a)
            x0, y0 = pos.astype(int)
            y1, y2 = max(0, y0 - 1), min(env.cfg.grid_size, y0 + 2)
            x1, x2 = max(0, x0 - 1), min(env.cfg.grid_size, x0 + 2)
            score = float(uncertainty[y1:y2, x1:x2].sum() + 0.15 * p[y1:y2, x1:x2].sum())
            if (int(x0), int(y0)) in occupied_next:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_action = int(a)
        next_pos = env.positions[i] if best_action == 8 else env.positions[i] + env.uav_step_vector(best_action)
        occupied_next.add((int(next_pos[0]), int(next_pos[1])))
        actions.append(best_action)
    return np.asarray(actions, dtype=int)


def run_wang_heuristic_demo(seed: int = 7) -> dict:
    """Run a non-learning demo of Wang-style TPM search."""
    env = WangSearchInterferenceEnv(WangConfig(seed=seed))
    history = []
    for _ in range(env.cfg.search_steps):
        actions = uncertainty_search_policy(env)
        result = env.step_search(actions)
        history.append(
            {
                "step": env.t,
                "coverage": result["coverage"],
                "reward": result["reward"],
                "positions": env.positions.copy().tolist(),
                "actions": actions.tolist(),
            }
        )
        if result["done"]:
            break
    return {
        "paper": "Wang et al. 2025",
        "experiment_kind": "heuristic_core_mechanism_demo",
        "note": "Uses an uncertainty search policy; not a full MAPPO reproduction.",
        "coverage": env.coverage(),
        "steps": env.t,
        "targets": env.targets.tolist(),
        "history": history,
    }


def run_wang_demo(seed: int = 7) -> dict:
    """Backward-compatible alias for the heuristic demo."""
    return run_wang_heuristic_demo(seed)
