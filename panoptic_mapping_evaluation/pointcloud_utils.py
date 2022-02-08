from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_labeled_pointcloud(pcd_file_path: Path):
    
    ply_data = PlyData.read(str(pcd_file_path))

    x = np.array(ply_data["vertex"].data["x"])
    y = np.array(ply_data["vertex"].data["y"])
    z = np.array(ply_data["vertex"].data["z"])
    points = np.column_stack((x, y, z))
    labels = np.array(ply_data["vertex"].data["label"])

    return points, labels


def labeled_pointcloud_to_grid(points: np.ndarray, labels: np.ndarray):
    """Convert labeled pointcloud to semantic grid"""
    # Convert all points to voxel coordinates
    points_voxel_coords = np.floor(points).astype(np.int32)

    # Aggregate all points belong to the same voxels and their inverse index
    non_empty_voxels, inv_idx = np.unique(points_voxel_coords, return_inverse=True, axis=0)

    # Initialize grid
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
