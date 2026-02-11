import habitat_sim
import magnum as mn
import numpy as np


def update_rotations(
    stationary: np.ndarray, new_rotations: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    _, m = stationary.shape
    nonstat = ~stationary
    cols = np.arange(m)
    left_idx = np.where(nonstat, cols, -1)
    last_idx = np.maximum.accumulate(left_idx, axis=1)
    idx_clipped = np.clip(last_idx, 0, m - 1)
    gathered = np.take_along_axis(new_rotations, idx_clipped, axis=1)
    carried = np.where(last_idx == -1, rotations[:, None], gathered)
    return np.where(stationary, carried, new_rotations)


def from_quat_to_angle(quat: mn.Quaternion) -> float:
    angle, axis = habitat_sim.utils.common.quat_to_angle_axis(quat)
    try:
        assert axis[0] == 0 and axis[2] == 0, (
            f"Only rotation around y-axis is supported, got axis: {axis}, angle: {angle}"
        )
    except AssertionError as _:
        assert angle == 0, (
            f"Angle should be zero if axis is not aligned with y-axis, got angle: {angle}"
        )
        return 0
    return angle * axis[1]
