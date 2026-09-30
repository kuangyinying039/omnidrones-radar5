"""Distributed constant-velocity target tracking primitives.

The estimator is deliberately independent of the environment and policy.  It
therefore can be validated before it is used as an observation or reward
source by MARL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _symmetrize_positive(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _stable_inverse_spd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Invert an SPD matrix through its spectrum instead of fragile LU."""
    matrix = _symmetrize_positive(matrix, floor=floor)
    values, vectors = np.linalg.eigh(matrix)
    ceiling = max(float(np.max(values)), floor) / floor
    inverse_values = 1.0 / np.clip(values, floor, ceiling)
    return (vectors * inverse_values) @ vectors.T


@dataclass
class TrackState:
    """One UAV's belief about one target with state ``[x, y, vx, vy]``."""

    target_id: int
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: int = -1
    last_seen_time: int = -1
    existence_probability: float = 0.0
    initialized: bool = False

    @classmethod
    def uninitialized(cls, target_id: int) -> "TrackState":
        return cls(
            target_id=target_id,
            mean=np.zeros(4, dtype=float),
            covariance=np.diag([1e3, 1e3, 1e2, 1e2]).astype(float),
        )

    def copy(self) -> "TrackState":
        return TrackState(
            self.target_id,
            self.mean.copy(),
            self.covariance.copy(),
            self.timestamp,
            self.last_seen_time,
            self.existence_probability,
            self.initialized,
        )

    def initialize(
        self,
        measurement_xy: np.ndarray,
        timestamp: int,
        measurement_variance: float,
        velocity_variance: float,
    ) -> None:
        self.mean[:] = [float(measurement_xy[0]), float(measurement_xy[1]), 0.0, 0.0]
        self.covariance = np.diag(
            [measurement_variance, measurement_variance, velocity_variance, velocity_variance]
        ).astype(float)
        self.timestamp = int(timestamp)
        self.last_seen_time = int(timestamp)
        self.existence_probability = 1.0
        self.initialized = True

    def predict(self, dt: float, process_acceleration_variance: float) -> None:
        if not self.initialized:
            return
        dt = max(float(dt), 1e-8)
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        g = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt], [dt, 0.0], [0.0, dt]])
        q = g @ (np.eye(2) * max(float(process_acceleration_variance), 0.0)) @ g.T
        self.mean = f @ self.mean
        self.covariance = _symmetrize_positive(f @ self.covariance @ f.T + q)

    def update_xy(
        self,
        measurement_xy: np.ndarray,
        timestamp: int,
        measurement_variance: float,
        velocity_variance: float = 4.0,
    ) -> float:
        """Apply a Cartesian position measurement and return information gain."""
        measurement = np.asarray(measurement_xy, dtype=float).reshape(2)
        variance = max(float(measurement_variance), 1e-8)
        if not self.initialized:
            self.initialize(measurement, timestamp, variance, velocity_variance)
            return 0.0
        prior_logdet = self.position_logdet()
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = np.eye(2) * variance
        innovation_covariance = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(innovation_covariance, h @ self.covariance).T
        self.mean = self.mean + gain @ (measurement - h @ self.mean)
        # Joseph form preserves positive semi-definiteness under rounding.
        identity = np.eye(4)
        residual = identity - gain @ h
        self.covariance = _symmetrize_positive(
            residual @ self.covariance @ residual.T + gain @ r @ gain.T
        )
        self.timestamp = int(timestamp)
        self.last_seen_time = int(timestamp)
        self.existence_probability = min(1.0, self.existence_probability + 0.2)
        return max(0.0, prior_logdet - self.position_logdet())

    def age(self, now: int, maximum: int) -> float:
        return float(maximum if self.timestamp < 0 else min(max(now - self.timestamp, 0), maximum))

    def position_logdet(self) -> float:
        if not self.initialized:
            return float("inf")
        sign, value = np.linalg.slogdet(self.covariance[:2, :2])
        return float(value if sign > 0 else np.log(1e-16))

    def position_trace(self) -> float:
        return float(np.trace(self.covariance[:2, :2]))


@dataclass
class TrackState3D(TrackState):
    """One UAV's 3-D target belief ``[x, y, z, vx, vy, vz]``."""

    @classmethod
    def uninitialized(cls, target_id: int) -> "TrackState3D":
        return cls(
            target_id=target_id,
            mean=np.zeros(6, dtype=float),
            covariance=np.diag([1e3, 1e3, 1e3, 1e2, 1e2, 1e2]).astype(float),
        )

    def copy(self) -> "TrackState3D":
        return TrackState3D(
            self.target_id,
            self.mean.copy(),
            self.covariance.copy(),
            self.timestamp,
            self.last_seen_time,
            self.existence_probability,
            self.initialized,
        )

    def initialize(
        self,
        measurement_xyz: np.ndarray,
        timestamp: int,
        measurement_variance: float,
        velocity_variance: float,
    ) -> None:
        measurement = np.asarray(measurement_xyz, dtype=float).reshape(3)
        self.mean[:] = [*measurement, 0.0, 0.0, 0.0]
        self.covariance = np.diag(
            [measurement_variance] * 3 + [velocity_variance] * 3
        ).astype(float)
        self.timestamp = int(timestamp)
        self.last_seen_time = int(timestamp)
        self.existence_probability = 1.0
        self.initialized = True

    def predict(self, dt: float, process_acceleration_variance: float) -> None:
        if not self.initialized:
            return
        dt = max(float(dt), 1e-8)
        identity3 = np.eye(3)
        f = np.block([[identity3, identity3 * dt], [np.zeros((3, 3)), identity3]])
        g = np.vstack([identity3 * (0.5 * dt * dt), identity3 * dt])
        q = g @ (identity3 * max(float(process_acceleration_variance), 0.0)) @ g.T
        self.mean = f @ self.mean
        self.covariance = _symmetrize_positive(f @ self.covariance @ f.T + q)

    def update_xyz(
        self,
        measurement_xyz: np.ndarray,
        timestamp: int,
        measurement_variance: float,
        velocity_variance: float = 4.0,
    ) -> float:
        """Apply a Cartesian 3-D position measurement and return information gain."""
        measurement = np.asarray(measurement_xyz, dtype=float).reshape(3)
        variance = max(float(measurement_variance), 1e-8)
        if not self.initialized:
            self.initialize(measurement, timestamp, variance, velocity_variance)
            return 0.0
        prior_logdet = self.position_logdet()
        h = np.hstack([np.eye(3), np.zeros((3, 3))])
        r = np.eye(3) * variance
        innovation_covariance = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(innovation_covariance, h @ self.covariance).T
        self.mean = self.mean + gain @ (measurement - h @ self.mean)
        identity = np.eye(6)
        residual = identity - gain @ h
        self.covariance = _symmetrize_positive(
            residual @ self.covariance @ residual.T + gain @ r @ gain.T
        )
        self.timestamp = int(timestamp)
        self.last_seen_time = int(timestamp)
        self.existence_probability = min(1.0, self.existence_probability + 0.2)
        return max(0.0, prior_logdet - self.position_logdet())

    def update_xy(self, *args, **kwargs) -> float:
        raise TypeError("TrackState3D requires update_xyz(), not update_xy()")

    def position_logdet(self) -> float:
        if not self.initialized:
            return float("inf")
        sign, value = np.linalg.slogdet(self.covariance[:3, :3])
        return float(value if sign > 0 else np.log(1e-16))

    def position_trace(self) -> float:
        return float(np.trace(self.covariance[:3, :3]))


@dataclass(frozen=True)
class TrackMessage:
    source_uav: int
    target_id: int
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: int
    last_seen_time: int
    existence_probability: float

    @classmethod
    def from_track(cls, source_uav: int, track: TrackState) -> "TrackMessage":
        if not track.initialized:
            raise ValueError("Cannot transmit an uninitialized track")
        return cls(
            int(source_uav), track.target_id, track.mean.copy(), track.covariance.copy(),
            track.timestamp, track.last_seen_time, track.existence_probability,
        )


def covariance_intersection(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    grid_size: int = 101,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse correlated estimates conservatively with determinant-optimal CI."""
    p1 = _symmetrize_positive(first_covariance)
    p2 = _symmetrize_positive(second_covariance)
    info1 = _stable_inverse_spd(p1)
    info2 = _stable_inverse_spd(p2)
    if p1.shape == (6, 6) and p2.shape == (6, 6):
        # 3-D pursuit only. Keep the same weight grid and determinant criterion,
        # but invert only the winning matrix rather than every improving weight.
        weights = np.linspace(0.0, 1.0, max(int(grid_size), 2))
        information = weights[:, None, None]*info1 + (1.0-weights[:, None, None])*info2
        signs, objectives = np.linalg.slogdet(information)
        valid = signs > 0
        if not np.any(valid):
            raise np.linalg.LinAlgError("Covariance intersection received invalid covariance matrices")
        index = int(np.argmax(np.where(valid, objectives, -np.inf)))
        weight = float(weights[index])
        covariance = _stable_inverse_spd(information[index])
        mean = covariance @ (weight*info1@first_mean+(1.0-weight)*info2@second_mean)
        return mean, _symmetrize_positive(covariance), weight
    best = None
    for weight in np.linspace(0.0, 1.0, max(int(grid_size), 2)):
        fused_info = weight * info1 + (1.0 - weight) * info2
        sign, objective = np.linalg.slogdet(fused_info)
        if sign <= 0:
            continue
        # Minimizing det(P) is equivalent to maximizing det(information).
        if best is None or objective > best[0]:
            covariance = _stable_inverse_spd(fused_info)
            mean = covariance @ (
                weight * info1 @ first_mean + (1.0 - weight) * info2 @ second_mean
            )
            best = (float(objective), mean, covariance, float(weight))
    if best is None:
        raise np.linalg.LinAlgError("Covariance intersection received invalid covariance matrices")
    return best[1], _symmetrize_positive(best[2]), best[3]


def freshness_score(
    age: float,
    tau: float,
    link_confidence: float = 1.0,
    floor: float = 0.05,
) -> float:
    """Return a bounded confidence score for delayed information.

    ``age`` is measured in environment steps and ``link_confidence`` is the
    current directed-link quality in ``[0, 1]``.  The score is deliberately
    independent of the state estimate so it cannot leak simulator truth.
    """
    tau = max(float(tau), 1e-8)
    floor = float(np.clip(floor, 1e-6, 1.0))
    link_confidence = float(np.clip(link_confidence, 0.0, 1.0))
    return float(np.clip(np.exp(-max(float(age), 0.0) / tau) * link_confidence, floor, 1.0))


def freshness_aware_covariance_intersection(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
    *,
    first_age: float,
    second_age: float,
    freshness_tau: float,
    second_link_confidence: float = 1.0,
    confidence_floor: float = 0.05,
    grid_size: int = 101,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Conservative CI after age/link-dependent covariance inflation.

    Freshness is incorporated by inflating each covariance before standard
    covariance intersection.  This preserves CI's no-cross-covariance
    requirement while preventing a numerically sharp but stale remote track
    from dominating a newer estimate.  The local estimate has unit link
    confidence; the received estimate is additionally discounted by the
    current directed-link confidence.
    """
    first_confidence = freshness_score(
        first_age, freshness_tau, 1.0, confidence_floor
    )
    second_confidence = freshness_score(
        second_age, freshness_tau, second_link_confidence, confidence_floor
    )
    inflated_first = _symmetrize_positive(first_covariance) / first_confidence
    inflated_second = _symmetrize_positive(second_covariance) / second_confidence
    return covariance_intersection(
        first_mean,
        inflated_first,
        second_mean,
        inflated_second,
        grid_size=grid_size,
    )


def fuse_track_with_message(
    local: TrackState,
    message: TrackMessage,
    *,
    now: int | None = None,
    freshness_tau: float | None = None,
    link_confidence: float = 1.0,
    confidence_floor: float = 0.05,
) -> float:
    """Fuse ``message`` into ``local`` and return the position information gain."""
    if not local.initialized:
        local.mean = message.mean.copy()
        local.covariance = message.covariance.copy()
        local.timestamp = message.timestamp
        local.last_seen_time = message.last_seen_time
        local.existence_probability = message.existence_probability
        local.initialized = True
        return 0.0
    prior = local.position_logdet()
    if now is not None and freshness_tau is not None:
        local.mean, local.covariance, _ = freshness_aware_covariance_intersection(
            local.mean,
            local.covariance,
            message.mean,
            message.covariance,
            first_age=max(int(now) - local.timestamp, 0),
            second_age=max(int(now) - message.timestamp, 0),
            freshness_tau=freshness_tau,
            second_link_confidence=link_confidence,
            confidence_floor=confidence_floor,
        )
    else:
        local.mean, local.covariance, _ = covariance_intersection(
            local.mean, local.covariance, message.mean, message.covariance
        )
    local.timestamp = max(local.timestamp, message.timestamp)
    local.last_seen_time = max(local.last_seen_time, message.last_seen_time)
    local.existence_probability = max(local.existence_probability, message.existence_probability)
    return max(0.0, prior - local.position_logdet())
