from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData, PlyElement

from evaluation.panoptic_mapping.constants import VOXEL_SIZE


def load_pointcloud(pcd_file_path: Path):
    ply_data = PlyData.read(str(pcd_file_path))
    x = np.array(ply_data["vertex"].data["x"])
    y = np.array(ply_data["vertex"].data["y"])
    z = np.array(ply_data["vertex"].data["z"])
    points = np.column_stack((x, y, z))
    return points


def make_occupancy_grid(
    points_gc: np.ndarray,
    max_voxel_coord: np.ndarray,
):
    """Generates a dense occupancy grid from a list of points in grid coordinates"""

    # Aggregate points belonging to the same cell
    non_empty_cells_grid_coords = np.unique(points_gc, axis=0)

    # Convert to indices
    indices = (
        non_empty_cells_grid_coords[:, 0],
        non_empty_cells_grid_coords[:, 1],
        non_empty_cells_grid_coords[:, 2],
    )

    # Initialize empty occupancy grid
    occupancy_grid = np.zeros(shape=max_voxel_coord, dtype=bool)

    # Set occupied cells
    occupancy_grid[indices] = True

    return occupancy_grid


def load_labeled_pointcloud(pcd_file_path: Path, return_colors=False):

    ply_data = PlyData.read(str(pcd_file_path))

    x = np.array(ply_data["vertex"].data["x"])
    y = np.array(ply_data["vertex"].data["y"])
    z = np.array(ply_data["vertex"].data["z"])
    points = np.column_stack((x, y, z))
    labels = np.array(ply_data["vertex"].data["label"])

    if return_colors:
        r = np.array(ply_data["vertex"].data["red"])
        g = np.array(ply_data["vertex"].data["green"])
        b = np.array(ply_data["vertex"].data["blue"])
        a = np.array(ply_data["vertex"].data["alpha"])
        colors = np.column_stack((r, g, b, a))
        return points, labels, colors
    else:
        return points, labels


def make_panoptic_grid(
    points_gc: np.ndarray,
    labels: np.ndarray,
    max_voxel_coord: np.ndarray,
):
    # Aggregate all points belong to the same voxels and their inverse index
    non_empty_voxels, inv_idx = np.unique(points_gc, return_inverse=True, axis=0)

    # Initialize grid
    voxel_grid = np.zeros(shape=max_voxel_coord, dtype=np.int32)

    # Fill voxel grid
    for idx, voxel_coord in enumerate(non_empty_voxels):
        # Skip voxels which are out of range
        if np.any(voxel_coord < [0, 0, 0]) or np.any(voxel_coord >= max_voxel_coord):
            continue
        # Grab the labels of all points belonging to the current voxel
        point_labels = labels[inv_idx == idx]
        # Compute the one with the highest number of occurrences
        voxel_label = np.argmax(np.bincount(point_labels))
        # Assign it to the voxel in the result voxel grid
        voxel_grid[voxel_coord[0], voxel_coord[1], voxel_coord[2]] = voxel_label

    return voxel_grid


def get_world_to_grid_transform(points: np.ndarray, voxel_size: float = VOXEL_SIZE):
    """Compute the world to grid transform for the given pointcloud

    The world to grid transform is an affine transform which moves all the points to the
    first octant and rescales them by the voxel size so that floor(T * points) returns
    the voxel coordinates of each points.
    """
    t = np.floor(np.min(points, axis=0) - voxel_size).reshape(-1, 1)

    T_G_W = np.block(
        [
            [np.identity(3) / voxel_size, -1 * t / voxel_size],
            [np.zeros((1, 3)), np.ones(1)],
        ]
    )
    return T_G_W


def save_labeled_pointcloud(
    pcd_file_path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    binary: Optional[bool] = True,
):
    labeled_pointcloud_data = np.array(
        [
            tuple(row)
            for row in np.concatenate((points, colors, labels.reshape(-1, 1)), axis=1)
        ],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("alpha", np.uint8),
            ("label", np.uint32),
        ],
    )

    # Export the pointcloud as ply
    ply_element = PlyElement.describe(labeled_pointcloud_data, "vertex")
    with pcd_file_path.open("wb") as f:
        PlyData([ply_element], text=not binary).write(f)


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


def convert_points_to_grid_coordinates(
    points: np.ndarray,
    w2g_transform: np.ndarray,
    max_grid_coord=None,
):
    # Transform points to the grid coordinate frame
    points_g = transform_points(points, w2g_transform)

    # Convert points to grid coordinates
    grid_coordinates = np.floor(points_g)

    # Clamp to the grid boundaries (only first octant)
    grid_coordinates = np.clip(grid_coordinates, a_min=(0, 0, 0), a_max=max_grid_coord)

    return grid_coordinates.astype(np.int32)
