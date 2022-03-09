import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import KDTree
from tqdm import tqdm

import evaluation.panoptic_mapping.pointcloud as pcd_utils
from utils.pano_seg import match_and_remap_panoptic_labels
from utils.visualization import colorize_panoptic_labels
from utils.common import NYU40_COLOR_PALETTE, NYU40_IGNORE_LABEL


def compute_vertex_map(depth_map, depth_intrinsic):
    cx, cy = depth_intrinsic[0:2, 2]
    fx_inv, fy_inv = 1 / depth_intrinsic[((0, 1), (0, 1))]
    u, v = np.meshgrid(
        np.arange(0, depth_map.shape[1]),
        np.arange(0, depth_map.shape[0]),
    )
    vertex_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
    vertex_map[:, :, 2] = depth_map
    vertex_map[:, :, 0] = (u.astype(np.float32) - cx) * fx_inv * depth_map
    vertex_map[:, :, 1] = (v.astype(np.float32) - cy) * fy_inv * depth_map
    return vertex_map


def main(
    scan_dir_path: Path,
    out_dir_path: Optional[Path] = None,
):
    assert scan_dir_path.is_dir()

    if out_dir_path is None:
        out_dir_path = scan_dir_path / "panoptic_temp_cons"

    out_dir_path.mkdir(parents=True, exist_ok=True)

    pano_seg_dir_path = scan_dir_path / "panoptic"
    assert pano_seg_dir_path.is_dir()

    pose_dir_path = scan_dir_path / "pose"
    assert pose_dir_path.is_dir()

    depth_dir_path = scan_dir_path / "depth"
    assert depth_dir_path.is_dir()

    # Load intrinsic matrix
    intrinsic_dir_path = scan_dir_path / "intrinsic"
    assert intrinsic_dir_path.is_dir()
    intrinsic_color_file_path = intrinsic_dir_path / "intrinsic_color.txt"
    intrinsic_color = np.loadtxt(str(intrinsic_color_file_path))[0:3, 0:3]
    intrinsic_depth_file_path = intrinsic_dir_path / "intrinsic_depth.txt"
    intrinsic_depth = np.loadtxt(str(intrinsic_depth_file_path))[0:3, 0:3]

    # Load gt mesh and create raycast scene
    gt_mesh_file_path = scan_dir_path / (scan_dir_path.name + "_vh_clean_2.labels.ply")
    assert gt_mesh_file_path.is_file()
    gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_file_path))
    gt_mesh = o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(gt_mesh)

    # Load labeled pcd pointcloud
    gt_pano_pcd_file_path = scan_dir_path / (scan_dir_path.name + ".pointcloud.ply")
    gt_points, gt_labels = pcd_utils.load_labeled_pointcloud(gt_pano_pcd_file_path)

    # Create color map
    color_map = dict(
        zip(
            np.unique(gt_labels),
            colorize_panoptic_labels(np.unique(gt_labels), NYU40_COLOR_PALETTE)[0],
        )
    )

    # Initialize kdtree for label lookups
    kdtree = KDTree(data=gt_points)

    colored_pano_seg_dir_path = out_dir_path / "colored"
    colored_pano_seg_dir_path.mkdir(exist_ok=True)

    for pano_seg_file_path in tqdm(sorted(pano_seg_dir_path.glob("*.png"))):
        # Load panoptic segmentation
        pano_seg = np.array(
            Image.open(pano_seg_file_path).resize((640, 480), resample=Image.NEAREST)
        )
        if np.all(pano_seg == 0):
            raise ValueError(f"Frame {pano_seg_file_path.stem} is invalid!")

        # Load gt pose
        pose_file_path = pose_dir_path / (pano_seg_file_path.stem + ".txt")
        pose = np.loadtxt(str(pose_file_path))

        # Create pinhole camera rays
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(intrinsic_color),
            o3d.core.Tensor(np.linalg.inv(pose)),
            pano_seg.shape[1],
            pano_seg.shape[0],
        )

        # Compute true depth with raycasting
        ans = scene.cast_rays(rays)
        depth_rc = ans["t_hit"].numpy()

        # Set no depth points to 0 and create mask
        no_depth_mask = np.isposinf(depth_rc)
        depth_rc[no_depth_mask] = 0

        # Compute vertex map using the raycast depth
        vertex_map = compute_vertex_map(depth_rc, intrinsic_depth)

        # Convert vertex
        points = pcd_utils.transform_points(vertex_map.reshape(-1, 3), pose)

        # Look up labels in the kdtree
        _, nn_idxs = kdtree.query(points, workers=-1)

        # Get the corresponding labels
        proj_pano_seg = gt_labels[nn_idxs]
        proj_pano_seg = proj_pano_seg.reshape(pano_seg.shape[0], pano_seg.shape[1])

        # Set the label of the points with no depth to ignore label
        proj_pano_seg[no_depth_mask] = NYU40_IGNORE_LABEL

        # Remap ids to projected pano seg
        remapped_pano_seg = match_and_remap_panoptic_labels(
            proj_pano_seg,
            pano_seg,
            ignore_unmatched=True,
        )

        # Save the result
        Image.fromarray(remapped_pano_seg).save(out_dir_path / pano_seg_file_path.name)

        # Also save RGB rendered panoptic seg
        colored_pano_seg = np.zeros(
            (remapped_pano_seg.shape[0], remapped_pano_seg.shape[1], 3),
            dtype=np.uint8,
        )
        for id in np.unique(remapped_pano_seg):
            try:
                colored_pano_seg[remapped_pano_seg == id] = color_map[id]
            except KeyError:
                pass
        Image.fromarray(colored_pano_seg).save(
            colored_pano_seg_dir_path / pano_seg_file_path.name
        )


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Make sequence of panoptic maps temporally consistent.",
    )

    parser.add_argument(
        "scan_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to the scan directory.",
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
        type=lambda p: Path(p).absolute(),
        help="Path to the output dir.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.scan_dir,
        args.out_dir,
    )
