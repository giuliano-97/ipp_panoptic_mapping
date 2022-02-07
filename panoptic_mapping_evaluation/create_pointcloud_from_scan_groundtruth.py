import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from plyfile import PlyData, PlyElement


_SEMANTIC_LABELED_MESH_FILE_TEMPLATE = "{}_vh_clean_2.labels.ply"
_SEGS_FILE_TEMPLATE = "{}_vh_clean_2.0.010000.segs.json"
_AGGREGATION_FILE_TEMPLATE = "{}_vh_clean.aggregation.json"
_IGNORE_LABEL = 0
_NYU40_STUFF_CLASSES = [1, 2, 22]
_PANOPTIC_LABEL_DIVISOR = 1000


def _load_ply_mesh(mesh_file_path: Path):
    with mesh_file_path.open("rb") as f:
        mesh = PlyData.read(f)
    return mesh


def _is_thing(semantic_id: int):
    return semantic_id not in _NYU40_STUFF_CLASSES and semantic_id != _IGNORE_LABEL


def _get_segments_to_object_id_dict(seg_groups: List):
    segment_to_object_id = dict()
    for group in seg_groups:
        object_id = group["objectId"]
        for seg in group["segments"]:
            segment_to_object_id[seg] = object_id
    return segment_to_object_id


def create_point_cloud_from_scan_grountruth(
    scan_dir_path: Path,
    out_dir_path: Path,
):
    assert scan_dir_path.is_dir()
    scene_name = scan_dir_path.stem

    # Check all the necessary files are there
    semantic_mesh_file_path = (
        scan_dir_path / _SEMANTIC_LABELED_MESH_FILE_TEMPLATE.format(scene_name)
    )
    assert semantic_mesh_file_path.is_file()

    segs_file_path = scan_dir_path / _SEGS_FILE_TEMPLATE.format(scene_name)
    assert segs_file_path.is_file()

    aggregation_file_path = scan_dir_path / _AGGREGATION_FILE_TEMPLATE.format(
        scene_name
    )
    assert aggregation_file_path.is_file()

    if out_dir_path is None:
        out_dir_path = scan_dir_path
    else:
        assert out_dir_path.is_dir()

    # Load meshes
    semantic_mesh = _load_ply_mesh(semantic_mesh_file_path)

    # Load the over-segmentation
    with segs_file_path.open("r") as f:
        segs_dict = json.load(f)
    seg_indices = segs_dict["segIndices"]

    # Load the aggregation file
    with aggregation_file_path.open("r") as f:
        aggregation_dict = json.load(f)
    seg_groups = aggregation_dict["segGroups"]

    # Get mapping from segments to object id
    segments_to_object_id = _get_segments_to_object_id_dict(seg_groups)

    # Generate panoptic labels
    num_vertices = semantic_mesh["vertex"].count
    vertex_labels = np.array(semantic_mesh["vertex"].data["label"])
    panoptic_vertex_labels = np.zeros_like(vertex_labels, dtype=np.uint32)
    for idx in range(num_vertices):
        semantic_label = vertex_labels[idx]
        panoptic_vertex_labels[idx] = semantic_label * _PANOPTIC_LABEL_DIVISOR
        if _is_thing(semantic_label):
            seg_group = seg_indices[idx]
            instance_id = segments_to_object_id[seg_group]
            # Add 1 because object ids start at 0
            panoptic_vertex_labels[idx] += instance_id + 1

    # Prepare pointcloud data for export
    vertex_x = np.array(semantic_mesh["vertex"].data["x"])
    vertex_y = np.array(semantic_mesh["vertex"].data["y"])
    vertex_z = np.array(semantic_mesh["vertex"].data["z"])
    color_r = np.array(semantic_mesh["vertex"].data["red"])
    color_g = np.array(semantic_mesh["vertex"].data["green"])
    color_b = np.array(semantic_mesh["vertex"].data["blue"])
    color_a = np.array(semantic_mesh["vertex"].data["alpha"])
    pointcloud_data = []
    for idx in range(num_vertices):
        pointcloud_data.append(
            (
                vertex_x[idx],
                vertex_y[idx],
                vertex_z[idx],
                color_r[idx],
                color_g[idx],
                color_b[idx],
                color_a[idx],
                panoptic_vertex_labels[idx],
            )
        )

    labeled_pointcloud = np.array(
        pointcloud_data,
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
    ply_element = PlyElement.describe(labeled_pointcloud, "vertex")
    output_file_path = out_dir_path / (
        scene_name + "_vh_clean_2.pointcloud.ply"
    )
    with output_file_path.open("wb") as f:
        PlyData([ply_element], text=True).write(f)


def _parse_args():

    parser = argparse.ArgumentParser(
        description="""
        Create a pointcloud with panoptic labels from the 3D annotations of a ScanNet V2 scan.
        The resulting pointcloud will be saved as a .ply pointcloud in which the "label" propery
        of every vertex represents its panoptic label in the format SEMANTIC_ID * 1000 + INSTANCE_ID.
        """
    )

    parser.add_argument(
        "scan_dir_path",
        type=lambda p: Path(p).expanduser(),
        help="Path to the scan directory.",
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        type=lambda p: Path(p).expanduser(),
        help="Path to the directory where the labeled pointcloud should be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_point_cloud_from_scan_grountruth(
        args.scan_dir_path,
        args.out_dir,
    )
