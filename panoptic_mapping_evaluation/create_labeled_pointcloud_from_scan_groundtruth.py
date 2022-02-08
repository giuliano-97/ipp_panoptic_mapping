import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scannet_utils import create_labeled_pointcloud_from_scan_groundtruth


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

    if args.out_dir is None:
        out_dir_path = args.scan_dir
    else:
        out_dir_path = args.out_dir

    points, colors, labels = create_labeled_pointcloud_from_scan_groundtruth(
        args.scan_dir_path,
    )

    labeled_pointcloud = np.array(
        [tuple(row) for row in np.concatenate((points, colors, labels), axis=1)],
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
    output_file_path = out_dir_path / (out_dir_path.name + ".pointcloud.ply")
    with output_file_path.open("wb") as f:
        PlyData([ply_element], text=True).write(f)
