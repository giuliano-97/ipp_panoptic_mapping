from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData

from panoptic_mapping_evaluation.constants import VOXEL_SIZE


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
        colors = np.column_stack((r,g,b,a))
        return points, labels, colors
    else:
        return points, labels


def labeled_pointcloud_to_grid(
    points: np.ndarray,
    labels: np.ndarray,
    max_voxel_coord: Optional[np.ndarray] = None,
):
    """Convert labeled pointcloud to semantic grid"""
    # Convert all points to voxel coordinates
    points_voxel_coords = np.floor(points).astype(np.int32)

    # Aggregate all points belong to the same voxels and their inverse index
    non_empty_voxels, inv_idx = np.unique(
        points_voxel_coords, return_inverse=True, axis=0
    )

    # Initialize grid
    if max_voxel_coord is None:
        max_voxel_coord = np.ceil(np.max(points, axis=0)).astype(np.int32)
    voxel_grid = np.squeeze(np.zeros(np.append(max_voxel_coord, 1), dtype=np.int32))

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
            [np.identity(3) / voxel_size, -1 * t],
            [np.zeros(1, 3), np.ones(1)],
        ]
    )
    return T_G_W
