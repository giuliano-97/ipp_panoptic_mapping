import argparse
import logging
from pathlib import Path
from typing import Mapping, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

import panoptic_mapping_evaluation.pointcloud as pcd_utils
import panoptic_mapping_evaluation.utils as utils
import panoptic_mapping_evaluation.constants as constants
from common import NYU40_IGNORE_LABEL, PANOPTIC_LABEL_DIVISOR


logging.basicConfig(level=logging.INFO)


_OFFSET = 256**2
_NUM_CLASSES = 41


def _ids_to_counts(id_grid: np.ndarray) -> Mapping[int, int]:
    """Given a numpy array, a mapping from each unique entry to its count."""
    ids, counts = np.unique(id_grid, return_counts=True)
    return dict(zip(ids, counts))


def evaluate_mean_iou(
    gt_panoptic_grid: np.ndarray,
    pred_panoptic_grid: np.ndarray,
) -> Dict[str, float]:
    assert gt_panoptic_grid.shape == pred_panoptic_grid.shape

    iou_per_class = np.zeros(_NUM_CLASSES, dtype=np.float32)

    # We evaluate IoU only over class labels
    gt_semantic_grid = gt_panoptic_grid // PANOPTIC_LABEL_DIVISOR
    pred_semantic_grid = pred_panoptic_grid // PANOPTIC_LABEL_DIVISOR

    # Only evaluate over classes that appear in the GT and are not the ignore label
    valid_classes = np.unique(gt_semantic_grid)
    valid_classes = valid_classes[valid_classes != NYU40_IGNORE_LABEL]

    # Compute IoU for every valid class
    for class_id in valid_classes:
        gt_class_mask = gt_semantic_grid == class_id
        pred_class_mask = pred_semantic_grid == class_id

        intersection_area = np.count_nonzero(gt_class_mask & pred_class_mask)
        union_area = np.count_nonzero(gt_class_mask | pred_class_mask)

        iou_per_class[class_id] = intersection_area / union_area

    # Compute mean iou
    miou = np.mean(np.take(iou_per_class, valid_classes))

    return {constants.MIOU_KEY: miou}


def evaluate_panoptic_reconstruction_quality(
    gt_panoptic_grid: np.ndarray,
    pred_panoptic_grid: np.ndarray,
) -> Dict[str, float]:
    assert gt_panoptic_grid.shape == pred_panoptic_grid.shape

    iou_per_class = np.zeros(_NUM_CLASSES, dtype=np.float64)
    tp_per_class = np.zeros(_NUM_CLASSES, dtype=np.float64)
    fp_per_class = np.zeros(_NUM_CLASSES, dtype=np.float64)
    fn_per_class = np.zeros(_NUM_CLASSES, dtype=np.float64)

    gt_segment_areas = _ids_to_counts(gt_panoptic_grid)
    pred_segment_areas = _ids_to_counts(pred_panoptic_grid)

    # Combine the groundtruth and predicted labels. Dividing up the voxels
    # based on which groundtruth segment and which predicted segment they belong
    # to, this will assign a different 64-bit integer label to each choice
    # of (groundtruth segment, predicted segment), encoded as
    #   gt_panoptic_label * offset + pred_panoptic_label.
    intersection_id_grid = gt_panoptic_grid.astype(
        np.int64
    ) * _OFFSET + pred_panoptic_grid.astype(np.int64)

    intersection_areas = _ids_to_counts(intersection_id_grid)

    gt_matched = set()
    pred_matched = set()

    for intersection_id, intersection_area in intersection_areas.items():
        gt_panoptic_label = intersection_id // _OFFSET
        pred_panoptic_label = intersection_id % _OFFSET

        gt_class_id = gt_panoptic_label // PANOPTIC_LABEL_DIVISOR
        pred_class_id = pred_panoptic_label // PANOPTIC_LABEL_DIVISOR

        if gt_class_id != pred_class_id:
            continue

        if pred_class_id == NYU40_IGNORE_LABEL:
            continue

        union = (
            gt_segment_areas[gt_panoptic_label]
            + pred_segment_areas[pred_panoptic_label]
            - intersection_area
        )

        iou = intersection_area / union
        if iou > constants.TP_IOU_THRESHOLD:
            tp_per_class[gt_class_id] += 1
            iou_per_class[gt_class_id] += iou
            gt_matched.add(gt_panoptic_label)
            pred_matched.add(pred_panoptic_label)

    for gt_panoptic_label in gt_segment_areas:
        if gt_panoptic_label in gt_matched:
            continue
        class_id = gt_panoptic_label // PANOPTIC_LABEL_DIVISOR
        # Failing to detect a void segment is not a false negative.
        if class_id == NYU40_IGNORE_LABEL:
            continue
        fn_per_class[class_id] += 1

    # Count false positives for each category.
    for pred_panoptic_label in pred_segment_areas:
        if pred_panoptic_label in pred_matched:
            continue
        class_id = pred_panoptic_label // PANOPTIC_LABEL_DIVISOR
        if class_id == NYU40_IGNORE_LABEL:
            continue
        fp_per_class[class_id] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        srq_per_class = np.nan_to_num(iou_per_class / tp_per_class)
        rrq_per_class = np.nan_to_num(
            tp_per_class / (tp_per_class + 0.5 * fp_per_class + 0.5 * fn_per_class)
        )
    prq_per_class = np.multiply(srq_per_class, rrq_per_class)

    valid_classes_mask = np.not_equal(tp_per_class + fp_per_class + fn_per_class, 0)

    qualities_per_class = np.row_stack((prq_per_class, srq_per_class, rrq_per_class))

    counts_per_class = np.row_stack((tp_per_class, fp_per_class, fn_per_class))

    prq, srq, rrq = np.mean(
        qualities_per_class[:, valid_classes_mask],
        axis=1,
    )

    tp, fp, fn = np.sum(
        counts_per_class[:, valid_classes_mask],
        axis=1,
    )

    return {
        constants.PRQ_KEY: prq,
        constants.SRQ_KEY: srq,
        constants.RRQ_KEY: rrq,
        constants.TP_KEY: tp,
        constants.FP_KEY: fp,
        constants.FN_KEY: fn,
    }


def evaluate_run(
    run_dir_path: Path,
    gt_pointcloud_file_path: Path,
) -> pd.DataFrame:
    assert run_dir_path.is_dir()
    assert gt_pointcloud_file_path.is_file()

    # Load the gt pointcloud with the labels
    points, labels = pcd_utils.load_labeled_pointcloud(gt_pointcloud_file_path)

    # Compute the points to grid transformation
    T_G_W = pcd_utils.get_world_to_grid_transform(
        points, voxel_size=constants.VOXEL_SIZE
    )

    # Transform and voxelize the groundtruth pointcloud
    gt_panoptic_grid = pcd_utils.make_panoptic_grid(
        points=utils.transform_points(points, T_G_W),
        labels=labels,
    )

    metrics_data = []

    # Comput the metrics for every map
    for pred_pointcloud_file_path in sorted(run_dir_path.glob("*.pointcloud.ply")):

        name = pred_pointcloud_file_path.name.rstrip(".pointcloud.ply")
        logging.info(f"Evaluating {name}")

        # Load the pointcloud
        pred_points, pred_labels = pcd_utils.load_labeled_pointcloud(
            pred_pointcloud_file_path
        )

        coverage_pcd_file_path = pred_pointcloud_file_path.parent / (
            name + ".coverage.ply"
        )
        if coverage_pcd_file_path.is_file():
            coverage_points = pcd_utils.load_pointcloud(coverage_pcd_file_path)
            coverage_grid = pcd_utils.make_occupancy_grid(
                points=utils.transform_points(coverage_points, T_G_W),
                max_voxel_coord=gt_panoptic_grid.shape,
            )
        else:
            coverage_grid = np.ones(gt_panoptic_grid.shape, dtype=bool)

        # Make the prediction panoptic grid
        pred_panoptic_grid = pcd_utils.make_panoptic_grid(
            points=utils.transform_points(pred_points, T_G_W),
            labels=pred_labels,
            max_voxel_coord=gt_panoptic_grid.shape,
        )

        # Create new metrics data entry
        metrics_data_entry = {
            "FrameID": pred_pointcloud_file_path.name.rstrip(".pointcloud.ply")
        }

        # Set the label of all voxels which have not been observed to the ignore label
        gt_panoptic_grid_covered = np.where(
            coverage_grid, gt_panoptic_grid, NYU40_IGNORE_LABEL
        )

        # Add PRQ, RRQ, SRQ, TP, FP, FN
        metrics_data_entry.update(
            evaluate_panoptic_reconstruction_quality(
                gt_panoptic_grid_covered,
                pred_panoptic_grid,
            )
        )

        # Add mIoU
        metrics_data_entry.update(
            evaluate_mean_iou(
                gt_panoptic_grid_covered,
                pred_panoptic_grid,
            )
        )

        # Append to list of metrics data entries
        metrics_data.append(metrics_data_entry)

    metrics_df = pd.DataFrame(metrics_data).set_index("FrameID").sort_index()
    return metrics_df


def main(
    run_dir_path: Path,
    gt_pointcloud_file_path: Path,
):
    assert run_dir_path.is_dir()
    assert gt_pointcloud_file_path.is_file()

    # Put metrics data into a dataframe
    metrics_df = evaluate_run(
        run_dir_path,
        gt_pointcloud_file_path,
    )

    # Metrics
    if metrics_df is None:
        logging.warning(f"{run_dir_path.name} could not be evaluated!")
        exit(1)
    else:
        metrics_file_path = run_dir_path / "metrics.csv"
        metrics_df.to_csv(str(metrics_file_path))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate panoptic reconstruction quality."
    )

    parser.add_argument(
        "run_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to run to be evaluated",
    )

    parser.add_argument(
        "gt_pointcloud_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the grountruth pointcloud to evaluate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        args.run_dir_path,
        args.gt_pointcloud_file_path,
    )
