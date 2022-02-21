import argparse
from pathlib import Path

import pointcloud as pcd_utils
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

    labeled_pointcloud_file_path = out_dir_path / (
        out_dir_path.name + ".pointcloud.ply"
    )

    pcd_utils.save_labeled_pointcloud(
        labeled_pointcloud_file_path,
        points,
        labels,
        colors,
    )
