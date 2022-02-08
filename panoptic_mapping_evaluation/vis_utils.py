import matplotlib as mpl
import numpy as np
import open3d as o3d

from panoptic_mapping_evaluation.constants import VOXEL_SIZE


def visualize_voxelized_labeled_pointcloud(
    points: np.ndarray,
    labels: np.ndarray,
    voxel_size: float = VOXEL_SIZE,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    hsv_cmap = mpl.cm.get_cmap("hsv")
    l_max = np.max(labels)
    colors = np.array([hsv_cmap(l / l_max) for l in labels])[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size,
    )

    o3d.visualization.draw_geometries([voxel_grid])
