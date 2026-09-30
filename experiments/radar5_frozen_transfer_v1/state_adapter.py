import numpy as np
import torch


def omnidrones_state_to_radar5(state):
    """
    OmniDrones MultirotorBase.get_state() layout:

    [position(3),
     quaternion(4),
     world_linear_velocity(3),
     world_angular_velocity(3),
     heading(3),
     up(3),
     normalized_throttle(4)]

    Radar5 layout:

    [position(3),
     world_linear_velocity(3),
     quaternion_wxyz(4),
     body_angular_velocity(3)]
    """
    if isinstance(state, torch.Tensor):
        value = state.detach().cpu().numpy()
    else:
        value = np.asarray(state)

    if value.shape[-1] < 23:
        raise ValueError(f"expected at least 23 state values, got {value.shape}")

    output = np.concatenate(
        [
            value[..., 0:3],    # position
            value[..., 7:10],   # world linear velocity
            value[..., 3:7],    # quaternion
            value[..., 10:13],  # angular velocity
        ],
        axis=-1,
    )

    if not np.isfinite(output).all():
        raise ValueError("converted Radar5 state contains non-finite values")

    quaternion_norm = np.linalg.norm(output[..., 6:10], axis=-1)
    if np.any(quaternion_norm < 1e-6):
        raise ValueError("converted quaternion has zero norm")

    output[..., 6:10] /= quaternion_norm[..., None]
    return output
