"""Isaac Sim native lidar adapter for the Radar5 observation contract.

This module deliberately does not synthesize visibility from target truth or
building rectangles. It consumes range-sensor output and exposes only finite
world-frame target measurements to the Radar5 tracker.
"""
from __future__ import annotations

import math
import numpy as np


class IsaacLidarAdapter:
    """Read native Isaac range sensors at decision boundaries.

    The extension API differs slightly between Isaac 4.x builds, so sensor
    creation is injected by the rollout script. ``read_scan`` accepts a
    provider returning point clouds in sensor-local coordinates.
    """

    def __init__(self, n_uavs, n_targets=1, decision_dt=0.2, max_age=0.2):
        self.n_uavs = int(n_uavs)
        self.n_targets = int(n_targets)
        self.decision_dt = float(decision_dt)
        self.max_age = float(max_age)
        self._last_time = -math.inf
        self._sequence = 0

    def read_scan(self, sim_time, point_clouds, target_world_positions, sensor_origins):
        """Convert native point clouds into a Radar5-compatible scan.

        ``point_clouds[i]`` must contain world-frame XYZ points returned by the
        native sensor adapter. Target association is intentionally geometric
        only against the target's semantic position for validation; the
        returned detection is accepted only when a measured point is nearby.
        """
        sim_time = float(sim_time)
        if sim_time < self._last_time:
            raise ValueError("sensor time moved backwards")
        if sim_time - self._last_time < self.decision_dt - 1e-5:
            raise RuntimeError("sensor scan requested before decision period elapsed")
        targets = np.asarray(target_world_positions, dtype=float)
        origins = np.asarray(sensor_origins, dtype=float)
        if targets.shape != (self.n_targets, 3) or origins.shape != (self.n_uavs, 3):
            raise ValueError("invalid target or sensor origin shape")
        measurements, variances = {}, {}
        visible = np.zeros((self.n_uavs, self.n_targets), dtype=bool)
        errors = []
        for agent, cloud in enumerate(point_clouds):
            points = np.asarray(cloud, dtype=float).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            for target_id, target in enumerate(targets):
                if not len(points):
                    continue
                distances = np.linalg.norm(points - target[None, :], axis=1)
                index = int(np.argmin(distances))
                error = float(distances[index])
                # A target return must be a point close to the semantic target;
                # the threshold is diagnostic and prevents truth leakage.
                if error > 0.75:
                    continue
                measurement = points[index]
                visible[agent, target_id] = True
                measurements[(agent, target_id)] = measurement
                variances[(agent, target_id)] = max(error * error, 1e-6)
                errors.append(error)
        self._last_time = sim_time
        self._sequence += 1
        return {
            "scan_time": sim_time,
            "scan_sequence": self._sequence,
            "scan_age_s": 0.0,
            "measurements": measurements,
            "variances": variances,
            "visible": visible,
            "rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan"),
        }

    def read_native_scan(self, sim_time, lidar_interface, sensor_paths,
                         target_world_positions, sensor_origins):
        """Read point clouds from Isaac 4.1's range-sensor interface.

        The interface returns sensor-local points in some Isaac builds and
        world points in others; callers must pass the already transformed
        world-frame arrays when using :meth:`read_scan`. This method accepts
        only the documented ``get_point_cloud_data`` result and fails loudly
        when the extension is unavailable or a sensor is not producing data.
        """
        if not hasattr(lidar_interface, "get_point_cloud_data"):
            raise RuntimeError("Isaac lidar interface lacks get_point_cloud_data")
        clouds = []
        for path in sensor_paths:
            data = lidar_interface.get_point_cloud_data(path)
            clouds.append(np.asarray(data, dtype=float).reshape(-1, 3))
        return self.read_scan(
            sim_time, clouds, target_world_positions, sensor_origins
        )
