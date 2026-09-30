"""Feed measured Isaac Lab pursuer states into Radar5's sensor and graph path.

Only pursuer state is replaced. Target dynamics, geometric lidar surrogate,
dropout, tracking and communication remain the original Radar5 implementation.
The frozen actor receives only agent_observations, comm_adjacency and
hetero_graph from this bridge, never the environment's target truth fields.
"""

from __future__ import annotations

import numpy as np

from pursuit_evasion_env import PursuitEvasionEnv
from pursuit_kinematics import yaw_from_quaternion
from pursuit_safety import project_velocity
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


class IsaacPursuitObservationBridge:
    def __init__(self, cfg: QuadrotorPursuitConfig):
        if cfg.n_uavs != 3 or cfg.n_targets != 1:
            raise ValueError("frozen pursuit actor requires three pursuers and one target")
        self.env = QuadrotorPursuitEnv(cfg)

    def reset(self) -> dict:
        return self.env.reset()

    def sync_pursuers(self, states_xyzw: np.ndarray) -> None:
        """Ingest [xyz, world linear velocity, xyzw, body angular velocity]."""
        physical = np.asarray(states_xyzw, dtype=np.float64)
        if physical.shape != (3, 13) or not np.isfinite(physical).all():
            raise ValueError("Isaac pursuer states must be finite with shape (3, 13)")
        norms = np.linalg.norm(physical[:, 6:10], axis=1)
        if np.any(norms < 1e-6):
            raise ValueError("Isaac quaternion has zero norm")
        policy = physical.copy()
        normalized = physical[:, 6:10] / norms[:, None]
        policy[:, 6:10] = normalized[:, [3, 0, 1, 2]]
        env = self.env
        env.quadrotor_states[:] = policy
        env.positions = policy[:, :2].copy()
        env.altitudes = policy[:, 2].copy()
        env.vertical_velocities = policy[:, 5].copy()
        yaw = np.array([yaw_from_quaternion(q) for q in policy[:, 6:10]])
        env.headings = np.mod(np.rint(yaw / (np.pi / 4)).astype(int), 8)

    def observe(self) -> dict:
        return self.env.observe_search()

    def filter_references(self, world_velocities: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
        """Apply Radar5's policy-causal local velocity projection."""
        requested = np.asarray(world_velocities, dtype=np.float64)
        if requested.shape != (3, 3) or not np.isfinite(requested).all():
            raise ValueError("world velocity references must be finite with shape (3, 3)")
        corrected = np.zeros_like(requested)
        reasons = []
        for agent in range(3):
            if self.env.disabled_uavs[agent]:
                reasons.append(["disabled"])
                continue
            corrected[agent], agent_reasons = project_velocity(
                self.env, agent, requested[agent], self.env.cfg.decision_dt
            )
            reasons.append(agent_reasons)
        return corrected, reasons

    def advance(self) -> dict:
        """Advance Radar5 target/sensing once after 20 external physics steps."""
        env = self.env
        env.current_decision_dt = env.cfg.decision_dt
        env._sensor_time += env.cfg.decision_dt
        # The production Radar5 continuous step uses this exact parent call
        # after integrating its own dynamics. Hover placeholders prevent the
        # base search model from applying a second pursuer displacement.
        result = PursuitEvasionEnv.step_joint(
            env, np.full(env.cfg.n_uavs, 8, dtype=np.int64)
        )
        return result
