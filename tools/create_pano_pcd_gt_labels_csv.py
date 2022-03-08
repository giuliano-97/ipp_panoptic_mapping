import argparse
from collections import defaultdict
from csv import DictWriter
from email.policy import default
from pathlib import Path

import numpy as np

import evaluation.panoptic_mapping.pointcloud as pcd_utils
from utils.visualization import colorize_panoptic_labels
from utils.common import (
    NYU40_COLOR_PALETTE,
    NYU40_CLASS_IDS_TO_NAMES,
    NYU40_CLASS_IDS_TO_SIZES,
    NYU40_IGNORE_LABEL,
    NYU40_STUFF_CLASSES,
    NYU40_THING_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
)

_FIELDS = [
    "InstanceID",
    "ClassID",
    "PanopticID",
    "R",
    "G",
    "B",
    "Name",
    "Size",
]

_INSTANCE_LABEL_CODE = 1
_BACKGROUND_LABELS_CODE = 2


def main(
    pano_pcd_file_path: Path,
):
    assert pano_pcd_file_path.is_file()
    out_dir_path = pano_pcd_file_path.parent
    out_file_path = out_dir_path / (out_dir_path.name + "_labels.csv")

    # Load labels from labeled pcd
    labels = np.unique(pcd_utils.load_labeled_pointcloud(pano_pcd_file_path)[1]) 

    # Compute colors
    colors, _ = colorize_panoptic_labels(labels, NYU40_COLOR_PALETTE)

    # Create csv labels file
    num_instances_per_class = defaultdict(int)
    with open(str(out_file_path), "w", newline="") as f:
        writer = DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()

        for label, color in zip(labels, colors):
            if label == NYU40_IGNORE_LABEL:
                continue
            class_id = label // PANOPTIC_LABEL_DIVISOR
            if class_id in NYU40_STUFF_CLASSES:
                instance_id = class_id
                label_code = _BACKGROUND_LABELS_CODE
            else:
                instance_id = label % PANOPTIC_LABEL_DIVISOR
                label_code = _INSTANCE_LABEL_CODE
            r, g, b = color
            name = NYU40_CLASS_IDS_TO_NAMES[class_id]
            if class_id in NYU40_THING_CLASSES:
                name += "_" + str(num_instances_per_class[class_id])
                num_instances_per_class[class_id] += 1
            size = NYU40_CLASS_IDS_TO_SIZES[class_id]
            writer.writerow(
                {
                    "InstanceID": instance_id,
                    "ClassID": class_id,
                    "PanopticID": label_code,
                    "R": r,
                    "G": g,
                    "B": b,
                    "Name": name,
                    "Size": size,
                }
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create CSV file mapping panoptic ids to objects.",
    )

    parser.add_argument(
        "pano_pcd_file",
        type=lambda p: Path(p).absolute(),
        help="Path to panoptic-labeled pointcloud",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.pano_pcd_file)
