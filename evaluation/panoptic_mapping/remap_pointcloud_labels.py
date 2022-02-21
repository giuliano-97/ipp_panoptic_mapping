import argparse
import csv
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pointcloud as pcd_utils
from utils.common import (
    PANOPTIC_LABEL_DIVISOR,
    NYU40_STUFF_CLASSES,
    NYU40_THING_CLASSES,
)


def _load_label_map(label_map_file_path: Path):
    label_map = {}
    with label_map_file_path.open("r") as f:
        dict_reader = csv.DictReader(f, fieldnames=["InstanceID", "ClassID"])

        for row in dict_reader:
            try:
                instance_id = int(row["InstanceID"])
                class_id = int(row["ClassID"])
                label_map.update({instance_id: class_id})
            except ValueError:
                continue
    return label_map


def main(dir_path: Path):
    assert dir_path.is_dir()

    def remap_labels_fn(panmap_file_path: Path):
        pointcloud_file_path = panmap_file_path.with_suffix(".pointcloud.ply")
        if not pointcloud_file_path.is_file():
            return
        label_map_file_path = panmap_file_path.with_suffix(".csv")
        if not label_map_file_path.is_file():
            return
        label_map = _load_label_map(label_map_file_path)

        # Skip if there is nothing to remap
        if len(label_map) == 0:
            return

        points, labels, colors = pcd_utils.load_labeled_pointcloud(
            pointcloud_file_path,
            return_colors=True,
        )

        # Remap the labels
        remapped_labels = labels.copy()
        for i, label in enumerate(labels):
            if label in label_map:
                class_id = label_map[label]
                if class_id in NYU40_THING_CLASSES:
                    remapped_labels[i] = label + class_id * PANOPTIC_LABEL_DIVISOR
            elif label in NYU40_STUFF_CLASSES:
                remapped_labels[i] = label * PANOPTIC_LABEL_DIVISOR

        # Save the pointcloud with the remapped labels
        pcd_utils.save_labeled_pointcloud(
            pointcloud_file_path,
            points,
            remapped_labels,
            colors,
        )

        # Remove the csv file
        os.remove(str(label_map_file_path))

    with ThreadPool(8) as p:
        p.map(remap_labels_fn, dir_path.glob("**/*.panmap"))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="",
    )

    parser.add_argument(
        "dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to directory containing pointclouds to remap.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.dir_path)
