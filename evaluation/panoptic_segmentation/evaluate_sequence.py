import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from panoptic_quality import PanopticQuality
from utils.common import NYU40_NUM_CLASSES
from utils.graphing import save_trend_lineplot

logging.basicConfig(level=logging.INFO)

_MAX_INSTANCES_PER_LABEL = 1000
_METRICS_NAMES = ["PQ", "SQ", "RQ", "TP", "FN", "FP"]


def _load_panoptic_segmentation(pano_seg_file_path: Path):
    pano_seg = np.array(Image.open(pano_seg_file_path))
    # Encode as 1-channel image
    if len(pano_seg.shape) == 2:
        return pano_seg
    # Encoded as one channel with dummy dimension
    elif pano_seg.shape[2] == 1:
        return np.squeeze(pano_seg)
    # Encoded as 2 channel
    elif pano_seg.shape[2] == 3:
        return pano_seg[:, :, 0] * _MAX_INSTANCES_PER_LABEL + pano_seg[:, :, 1]


def evaluate_sequence(
    pred_pano_seg_dir_path: Path,
    gt_pano_seg_dir_path: Path,
    metrics_log_interval: int,
    output_dir_path: Path,
):
    assert pred_pano_seg_dir_path.is_dir()
    assert gt_pano_seg_dir_path.is_dir()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    pq = PanopticQuality(
        num_classes=NYU40_NUM_CLASSES,
        ignored_label=0,
        max_instances_per_category=1000,
        offset=256 * 256,
    )

    pred_pano_seg_files = list(pred_pano_seg_dir_path.glob("*.png"))
    metrics_data = []
    for i, pred_pano_seg_file_path in enumerate(sorted(pred_pano_seg_files)):
        gt_pano_seg_file_path = gt_pano_seg_dir_path / pred_pano_seg_file_path.name
        if not gt_pano_seg_file_path.is_file():
            logging.warning(
                f"{gt_pano_seg_file_path.name} groundtruth not found. Skipped."
            )
        pred_pano_seg = _load_panoptic_segmentation(pred_pano_seg_file_path)
        gt_pano_seg = _load_panoptic_segmentation(gt_pano_seg_file_path)

        # Convert to tensor and evaluate
        pq.update_state(
            tf.convert_to_tensor(gt_pano_seg),
            tf.convert_to_tensor(pred_pano_seg),
        )

        if i % metrics_log_interval == 0 or i == len(pred_pano_seg_files) - 1:
            metrics_data_entry = dict(zip(_METRICS_NAMES, pq.result().numpy().tolist()))
            metrics_data_entry["FrameID"] = i + 1
            metrics_data.append(metrics_data_entry)

    # Pack metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Plot the metrics
    trend_plot_file_path = output_dir_path / "panoptic_segmentation_metrics.png"
    save_trend_lineplot(
        metrics_df,
        trend_plot_file_path,
    )

    # Save the metrics to file
    metrics_file_path = output_dir_path / "panoptic_segmentation_metrics.csv"
    metrics_df.set_index("FrameID").to_csv(metrics_file_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate panoptic quality for all the images in the target dir."
    )

    parser.add_argument(
        "pred_pano_seg_dir",
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "gt_pano_seg_dir",
        type=lambda p: Path(p).absolute(),
    )
    parser.add_argument(
        "--metrics_log_interval",
        type=int,
        default=5,
        help="Save partial results every n frames.",
    )

    parser.add_argument(
        "--output_dir",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_sequence(
        args.pred_pano_seg_dir,
        args.gt_pano_seg_dir,
        args.metrics_log_interval,
        args.output_dir,
    )
