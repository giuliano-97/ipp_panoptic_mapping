import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import time

from metrics.panoptic_quality import PanopticQuality

logging.basicConfig(level=logging.INFO)

_MAX_INSTANCES_PER_LABEL = 1000
_METRICS_NAMES = ["PQ", "SQ", "RQ", "TP", "FN", "FP"]


def _encode_panoptic(label_image: np.ndarray):
    return label_image[:, :, 0] * _MAX_INSTANCES_PER_LABEL + label_image[:, :, 1]


def eval_panoptic_quality(
    pred_labels_dir: str,
    gt_labels_dir: str,
    output_dir: str,
):
    assert os.path.isdir(pred_labels_dir), f"{pred_labels_dir} is not a valid dir."
    assert os.path.isdir(gt_labels_dir), f"{gt_labels_dir} is not a valid dir."
    os.makedirs(output_dir, exist_ok=True)

    pq = PanopticQuality(
        num_classes=41,
        ignored_label=0,
        max_instances_per_category=1000,
        offset=256 * 256,
    )

    pred_labels_files = glob.glob(f"{pred_labels_dir}/*.png")
    metrics_data = np.empty(
        shape=(len(pred_labels_files), len(_METRICS_NAMES)),
        dtype=np.float32,
    )
    for i, pred_labels_file in enumerate(sorted(pred_labels_files)):
        gt_labels_file = os.path.join(gt_labels_dir, os.path.basename(pred_labels_file))
        assert os.path.exists(
            gt_labels_file
        ), f"Ground truth labels file {gt_labels_file} not found!"
        pred_labels = _encode_panoptic(
            np.array(Image.open(pred_labels_file), dtype=np.int64)
        )
        gt_labels = np.array(Image.open(gt_labels_file), dtype=np.int64)

        # Convert to tensor and evaluate
        pq.update_state(
            tf.convert_to_tensor(gt_labels),
            tf.convert_to_tensor(pred_labels),
        )

        metrics_data[i, :] = pq.result().numpy()

        pq.reset_states()

    metrics_df = pd.DataFrame(
        metrics_data,
        columns=_METRICS_NAMES,
    )
    metrics_file_path = os.path.join(
        output_dir,
        os.path.basename(pred_labels_dir) + "_metrics.csv",
    )

    metrics_df.to_csv(metrics_file_path, index=False)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate panoptic quality for all the images in the target dir."
    )

    parser.add_argument(
        "--pred_labels_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--gt_labels_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    eval_panoptic_quality(
        args.pred_labels_dir,
        args.gt_labels_dir,
        args.output_dir,
    )
