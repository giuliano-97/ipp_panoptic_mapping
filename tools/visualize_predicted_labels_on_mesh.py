import argparse
from pathlib import Path

import open3d as o3d
import numpy as np

import evaluation.panoptic_mapping.pointcloud as pcd_utils
from utils.common import NYU40_COLOR_PALETTE, NYU40_IGNORE_LABEL
from utils.visualization import colorize_panoptic_labels
   


def visualize_predicted_labels_on_mesh(
    pred_pcd_file_path: Path,
    gt_mesh_file_path: Path,
):
    assert pred_pcd_file_path.is_file()
    assert gt_mesh_file_path.is_file()

    # Load the gt triangle mesh
    mesh = o3d.io.read_triangle_mesh(str(gt_mesh_file_path))

    # Get vertices
    gt_points = np.asarray(mesh.vertices)

    # Compute world to grid transform
    w2g_transform = pcd_utils.get_world_to_grid_transform(gt_points)

    # Convert to grid coordinates
    gt_points_gc = pcd_utils.convert_points_to_grid_coordinates(
        gt_points, w2g_transform
    )

    # Compute grid bounding box - just one point since it's in the first octant
    max_grid_coord = np.ceil(np.max(gt_points_gc, axis=0)).astype(np.int32) + 1

    # Load gt poincloud and labels
    pred_points, pred_labels = pcd_utils.load_labeled_pointcloud(pred_pcd_file_path)
    # Convert pred_points to grid coordinates
    pred_points_gc = pcd_utils.convert_points_to_grid_coordinates(
        pred_points, w2g_transform, max_grid_coord
    )
    
    # Create label grid
    pred_label_grid = pcd_utils.make_panoptic_grid(
        pred_points_gc, pred_labels, max_grid_coord,
    )

    # Look up the label for each vertex in the mesh
    pred_gt_labels = pred_label_grid[tuple(gt_points_gc.T)]

    # Generate color mapping
    colors, _ = colorize_panoptic_labels(pred_gt_labels, NYU40_COLOR_PALETTE)

    # Change the colors of the vertices 
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Display
    o3d.visualization.draw_geometries([mesh])


def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize predicted labels on mesh")

    parser.add_argument(
        "pred_pcd_file",
        type=lambda p: Path(p).absolute(),
        help="Path to the predicted labeled pointcloud.",
    )

    parser.add_argument(
        "gt_mesh_file",
        type=lambda p: Path(p).absolute(),
        help="Path to the groundtruth mesh file path.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_predicted_labels_on_mesh(
        args.pred_pcd_file,
        args.gt_mesh_file,
    )
