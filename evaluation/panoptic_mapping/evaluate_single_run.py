import argparse
import logging
from pathlib import Path

import pandas as pd

import pointcloud as pcd_utils
from pano_pcd_evaluator import PanopticPointcloudEvaluator
from utils.common import NYU40_NUM_CLASSES

logging.basicConfig(level=logging.INFO)


def evaluate_run(
    run_dir_path: Path,
    gt_pointcloud_file_path: Path,
) -> pd.DataFrame:
    assert run_dir_path.is_dir()
    assert gt_pointcloud_file_path.is_file()

    # Load the gt pointcloud with the labels
    gt_points, gt_labels = pcd_utils.load_labeled_pointcloud(gt_pointcloud_file_path)

    evaluator = PanopticPointcloudEvaluator(gt_points, gt_labels)

    metrics_data = []

    # Comput the metrics for every map
    for pred_pointcloud_file_path in sorted(run_dir_path.glob("*.pointcloud.ply")):

        name = pred_pointcloud_file_path.name.rstrip(".pointcloud.ply")
        logging.info(f"Evaluating {name}")

        # Load the pointcloud
        pred_points, pred_labels = pcd_utils.load_labeled_pointcloud(
            pred_pointcloud_file_path
        )

        # Make sure no invalid labels are there
        pred_labels[pred_labels > (NYU40_NUM_CLASSES - 1) * 1000] = 0

        coverage_pcd_file_path = pred_pointcloud_file_path.parent / (
            name + ".coverage.ply"
        )
        if coverage_pcd_file_path.is_file():
            coverage_points = pcd_utils.load_pointcloud(coverage_pcd_file_path)
        else:
            coverage_points = None

        metrics_data_entry = {
            "FrameID": int(pred_pointcloud_file_path.name.rstrip(".pointcloud.ply"))
        }

        metrics_data_entry.update(
            evaluator.evaluate(
                pred_points,
                pred_labels,
                coverage_points,
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
