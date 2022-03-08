import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

import evaluation.panoptic_mapping.pointcloud as pcd_utils
from utils.common import (
    NYU40_STUFF_CLASSES,
    NYU40_IGNORE_LABEL,
    PANOPTIC_LABEL_DIVISOR,
    NYU40_COLOR_PALETTE,
)
from utils.visualization import colorize_panoptic_labels

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


def main(
    scan_dir_path: Path,
    out_dir_path: Optional[Path],
):
    assert scan_dir_path.is_dir()

    if out_dir_path is None:
        out_dir_path = scan_dir_path
    else:
        out_dir_path = out_dir_path

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
    points, semantic_labels, original_colors = pcd_utils.load_labeled_pointcloud(
        semantic_mesh_file_path, return_colors=True
    )

    # Generate panoptic labels
    panoptic_labels = np.zeros_like(semantic_labels, dtype=np.uint32)
    object_id_to_instance_id = dict()
    next_valid_instance_id = 0
    for idx, semantic_label in enumerate(semantic_labels):
        panoptic_labels[idx] = semantic_label * PANOPTIC_LABEL_DIVISOR
        if _is_thing(semantic_label):
            object_id = segments_to_object_id[seg_indices[idx]]
            instance_id = object_id_to_instance_id.get(object_id, None)
            if instance_id is None:
                # Grab the next valid instance id
                while (
                    next_valid_instance_id == NYU40_IGNORE_LABEL
                    or next_valid_instance_id in NYU40_STUFF_CLASSES
                ):
                    next_valid_instance_id += 1
                instance_id = next_valid_instance_id
                object_id_to_instance_id[object_id] = instance_id
                next_valid_instance_id += 1

            # Add 1 because object ids start at 0
            panoptic_labels[idx] += instance_id

    labeled_pointcloud_file_path = out_dir_path / (
        out_dir_path.name + ".pointcloud.ply"
    )

    # Create new colors
    new_colors, _ = colorize_panoptic_labels(panoptic_labels, NYU40_COLOR_PALETTE)

    # Add alpha channel
    new_colors = np.insert(new_colors, 3, values=original_colors[:, 3], axis=1)

    pcd_utils.save_labeled_pointcloud(
        labeled_pointcloud_file_path,
        points,
        panoptic_labels,
        new_colors,
    )


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
    main(args.scan_dir_path, args.out_dir)
