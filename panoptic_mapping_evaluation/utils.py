import numpy as np


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError("tranform should be 4x4 matrix!")

    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError("points should be an Nx3 array!")

    # Homogenize and convert to columns
    points_h_t = np.transpose(np.c_[points, np.ones(points.shape[0])])

    # Apply the transformation
    tranformed_points_h_t = np.matmul(transform, points_h_t)

    # Dehomogenize, tranpose, and return
    return np.transpose(tranformed_points_h_t[:3, :])
