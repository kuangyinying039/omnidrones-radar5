"""World z-up / body FLU orientation helpers; quaternion order wxyz."""
import math
import numpy as np

def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )

def quaternion_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)

def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
