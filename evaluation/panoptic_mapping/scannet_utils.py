import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import pointcloud as pcd_utils
from utils.common import NYU40_STUFF_CLASSES, NYU40_IGNORE_LABEL, PANOPTIC_LABEL_DIVISOR

_SEMANTIC_LABELED_MESH_FILE_TEMPLATE = "{}_vh_clean_2.labels.ply"
_SEGS_FILE_TEMPLATE = "{}_vh_clean_2.0.010000.segs.json"
_AGGREGATION_FILE_TEMPLATE = "{}_vh_clean.aggregation.json"


def _is_thing(semantic_id: int):
    return semantic_id not in NYU40_STUFF_CLASSES and semantic_id != NYU40_IGNORE_LABEL


def _get_segments_to_object_id_dict(seg_groups: List):
    segment_to_object_id = dict()
    for group in seg_groups:
        object_id = group["objectId"]
        for seg in group["segments"]:
            segment_to_object_id[seg] = object_id
    return segment_to_object_id


def create_labeled_pointcloud_from_scan_groundtruth(
    scan_dir_path: Path,
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

    # Load the semantic mesh as a pointcloud with colors and labels
    points, colors, semantic_labels = pcd_utils.load_labeled_pointcloud(
        semantic_mesh_file_path,
        return_colors=True,
    )

    # Generate panoptic labels
    panoptic_labels = np.zeros_like(semantic_labels, dtype=np.uint32)
    for idx, semantic_label in enumerate(semantic_labels):
        panoptic_labels[idx] = semantic_label * PANOPTIC_LABEL_DIVISOR
        if _is_thing(semantic_label):
            seg_group = seg_indices[idx]
            instance_id = segments_to_object_id[seg_group]
            # Add 1 because object ids start at 0
            panoptic_labels[idx] += instance_id + 1

    return points, colors, panoptic_labels