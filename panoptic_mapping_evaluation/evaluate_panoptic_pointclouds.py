import argparse
import csv
from fileinput import filename
import json
import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common import NYU40_IGNORE_LABEL, NYU40_STUFF_CLASSES, NYU40_THING_CLASSES
from panoptic_mapping_evaluation.metrics import (
    mean_iou,
    panoptic_reconstruction_quality,
)

logging.basicConfig(level=logging.INFO)


_PANOPTIC_LABEL_DIVISOR = 1000
_VOXEL_SEGS_FILE_EXTENSION = ".voxel_segs.json"


def find_voxel_segs_files(pred_voxel_segs_dir_path: Path):
    return sorted(
        [
            p
            for p in pred_voxel_segs_dir_path.glob(f"*{_VOXEL_SEGS_FILE_EXTENSION}")
            if p.is_file()
        ]
    )


def decode_panoptic_label(panoptic_label):
    return (
        panoptic_label // _PANOPTIC_LABEL_DIVISOR,
        panoptic_label % _PANOPTIC_LABEL_DIVISOR,
    )


def aggregate_voxel_segs_by_class(voxel_segs):
    voxel_segs_per_class = dict()
    for segment in voxel_segs:
        label = segment["id"]
        if label == NYU40_IGNORE_LABEL:
            continue
        semantic_id, _ = decode_panoptic_label(label)
        if semantic_id not in voxel_segs_per_class:
            voxel_segs_per_class[semantic_id] = []
        voxel_indices = np.array(
            [tuple(s) for s in segment["voxels"]],
            dtype=[
                ("i", np.int32),
                ("j", np.int32),
                ("k", np.int32),
            ],
        )
        voxel_segs_per_class[semantic_id].append(voxel_indices)
    return voxel_segs_per_class


def load_voxel_segs_from_file(voxel_segs_file: Path):
    with voxel_segs_file.open("r") as j:
        voxel_segs = json.load(j)

    return voxel_segs


def load_instance_to_class_id_map(id_to_class_map_file_path: Path):
    instance_to_class_id_map = dict()
    with id_to_class_map_file_path.open("r") as f:
        reader = csv.DictReader(f, fieldnames=["InstanceID", "ClassID"])

        for row in reader:
            try:
                instance_to_class_id_map.update(
                    {int(row["InstanceID"]): int(row["ClassID"])}
                )
            except ValueError:
                pass
    return instance_to_class_id_map


def map_voxel_segs_ids_to_panoptic(
    voxel_segs: List[Dict],
    instance_id_to_class_map: Dict[int, int],
):
    for voxel_seg in voxel_segs:
        # Stuff segments
        if voxel_seg["id"] in NYU40_STUFF_CLASSES:
            voxel_seg["id"] *= _PANOPTIC_LABEL_DIVISOR
        # Instance segments
        else:
            # Ignore instance segments whose class is not known
            if voxel_seg["id"] not in instance_id_to_class_map:
                voxel_seg["id"] = NYU40_IGNORE_LABEL
            instance_id = voxel_seg["id"]
            class_id = instance_id_to_class_map[instance_id]
            # Ignore instance segments mapped to an invalid class
            if class_id not in NYU40_THING_CLASSES:
                voxel_seg["id"] = NYU40_IGNORE_LABEL
            # Normalize the instance id just in case
            voxel_seg["id"] = (
                class_id * _PANOPTIC_LABEL_DIVISOR
                + instance_id % _PANOPTIC_LABEL_DIVISOR
            )


def evaluate_panoptic_pointclouds(
    pred_voxel_segs_dir_path: Path,
    gt_voxel_segs_file_path: Path,
    map_ids_to_panoptic: bool = False,
    output_dir_path: Optional[Path] = None,
):
    assert pred_voxel_segs_dir_path.is_dir()
    assert gt_voxel_segs_file_path.is_file()

    if output_dir_path is not None:
        assert output_dir_path.exists()
    else:
        output_dir_path = pred_voxel_segs_dir_path

    # Load grountruth
    gt_voxel_segs_per_class = aggregate_voxel_segs_by_class(
        load_voxel_segs_from_file(gt_voxel_segs_file_path)
    )

    # Initialize list of metrics data entries
    metrics_data = []

    # Find all the segs files
    pred_voxel_segs_files = find_voxel_segs_files(pred_voxel_segs_dir_path)

    # Compute the metrics for every run
    for pred_voxel_segs_file in pred_voxel_segs_files:
        logging.info(f"Evaluating {pred_voxel_segs_file.name}")
        pred_voxel_segs = load_voxel_segs_from_file(pred_voxel_segs_file)

        # Map predicted voxel segs ids to panoptic format (semantic_id * 1000 + instance id)
        if map_ids_to_panoptic:
            instance_to_class_id_file_path = pred_voxel_segs_file.parent.joinpath(
                pred_voxel_segs_file.name.replace(_VOXEL_SEGS_FILE_EXTENSION, ".csv")
            )
            if not instance_to_class_id_file_path.is_file():
                logging.error(
                    "Instance to class ID map file not found for "
                    f"{pred_voxel_segs_file.name}. Skipped."
                )
                continue
            instance_to_class_id_map = load_instance_to_class_id_map(
                instance_to_class_id_file_path
            )
            map_voxel_segs_ids_to_panoptic(
                pred_voxel_segs, instance_to_class_id_map
            )

        pred_voxel_segs_per_class = aggregate_voxel_segs_by_class(pred_voxel_segs)
        metrics_data_entry = {
            "Name": pred_voxel_segs_file.name.rstrip(_VOXEL_SEGS_FILE_EXTENSION)
        }
        prq_metrics_dict = panoptic_reconstruction_quality(
            pred_voxel_segs, gt_voxel_segs_per_class
        )
        metrics_data_entry.update(prq_metrics_dict)
        metrics_data_entry.update(
            {"mIoU": mean_iou(pred_voxel_segs_per_class, gt_voxel_segs_per_class)}
        )
        metrics_data.append(metrics_data_entry)

    # Put metrics data into a dataframe
    metrics_df = pd.DataFrame(metrics_data)

    # Save the metrics to file
    metrics_file_path = pred_voxel_segs_dir_path / "mapping_metrics.csv"
    metrics_df.set_index("Name").to_csv(metrics_file_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate panoptic reconstruction quality."
    )

    parser.add_argument(
        "pred_voxel_segs_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to directory with *_voxel_segs.json files to be evaluated.",
    )

    parser.add_argument(
        "gt_voxel_segs_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the grountruth voxel segments to evaluate against.",
    )

    parser.add_argument(
        "--map_ids_to_panoptic",
        action="store_true",
        help="Remap predicted labels to the panoptic format before computing the metrics.",
    )

    parser.add_argument(
        "-o",
        "--output_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the output directory where the metrics file should be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_panoptic_pointclouds(
        args.pred_voxel_segs_dir_path,
        args.gt_voxel_segs_file_path,
        args.map_ids_to_panoptic,
        args.output_dir_path,
    )
