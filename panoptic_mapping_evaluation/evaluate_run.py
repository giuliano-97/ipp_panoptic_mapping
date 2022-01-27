import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common import NYU40_IGNORE_LABEL
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


def load_voxel_segs_from_file(voxel_segs_file: Path):
    with voxel_segs_file.open("r") as j:
        voxel_segs_list = json.load(j)

    voxel_segs = dict()
    for segment in voxel_segs_list:
        label = segment["id"]
        if label == NYU40_IGNORE_LABEL:
            continue
        semantic_id, _ = decode_panoptic_label(label)
        if semantic_id not in voxel_segs:
            voxel_segs[semantic_id] = []
        voxel_indices = np.array(
            [tuple(s) for s in segment["voxels"]],
            dtype=[
                ("i", np.int32),
                ("j", np.int32),
                ("k", np.int32),
            ],
        )
        voxel_segs[semantic_id].append(voxel_indices)
    return voxel_segs


def evaluate_run(
    pred_voxel_segs_dir_path: Path,
    gt_voxel_segs_file_path: Path,
    output_dir_path: Path = None,
):
    assert pred_voxel_segs_dir_path.is_dir()
    assert gt_voxel_segs_file_path.is_file()

    if output_dir_path is not None:
        assert output_dir_path.exists()
    else:
        output_dir_path = pred_voxel_segs_dir_path

    # Load grountruth
    gt_voxel_segs = load_voxel_segs_from_file(gt_voxel_segs_file_path)

    # Initialize list of metrics data entries
    metrics_data = []

    # Compute metrics for every result file
    pred_voxel_segs_files = find_voxel_segs_files(pred_voxel_segs_dir_path)

    # Compute the metrics for every run
    for pred_voxel_segs_file in pred_voxel_segs_files:
        logging.info(f"Evaluating {pred_voxel_segs_file.name}")
        pred_voxel_segs = load_voxel_segs_from_file(pred_voxel_segs_file)
        metrics_data_entry = {"Name": pred_voxel_segs_file.name.rstrip(_VOXEL_SEGS_FILE_EXTENSION)}
        prq_metrics_dict = panoptic_reconstruction_quality(
            pred_voxel_segs, gt_voxel_segs
        )
        metrics_data_entry.update(prq_metrics_dict)
        metrics_data_entry.update({"mIoU": mean_iou(pred_voxel_segs, gt_voxel_segs)})

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
        help="Path to directory with *.voxel_segs.json files to be evaluated.",
    )

    parser.add_argument(
        "gt_voxel_segs_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the grountruth voxel segments to evaluate against.",
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
    evaluate_run(
        args.pred_voxel_segs_dir_path,
        args.gt_voxel_segs_file_path,
        args.output_dir_path,
    )
