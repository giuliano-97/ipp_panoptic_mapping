import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from evaluate_single_run import evaluate_run


def _find_runs(runs_dir_path: Path) -> List[Path]:
    run_dirs = []
    for p in runs_dir_path.iterdir():
        if not p.is_dir():
            continue
        if len(list(p.glob("*.pointcloud.ply"))) > 0:
            run_dirs.append(p)
    
    return sorted(run_dirs, reverse=True)


def evaluate_and_compare_runs(
    gt_pointcloud_file_path: Path,
    runs_dir_path: Path,
    overwrite: bool,
):
    assert gt_pointcloud_file_path.is_file()

    run_dirs = _find_runs(runs_dir_path)
    if len(run_dirs) == 0:
        logging.warning(f"No runs to evaluate were found in {runs_dir_path.name}")

    cumulative_metrics_data = []

    for run_dir_path in run_dirs:
        logging.info(f"Evaluating run: {run_dir_path.name}")

        metrics_file_path = run_dir_path / "metrics.csv"
        if not overwrite and metrics_file_path.is_file():
            metrics_df = pd.read_csv(str(metrics_file_path), index_col="FrameID")
        else:
            metrics_df = evaluate_run(
                run_dir_path=run_dir_path,
                gt_pointcloud_file_path=gt_pointcloud_file_path,
            )

            if metrics_df is None:
                logging.warning(
                    f"Run {run_dir_path.name} could not be evaluated. Skipped."
                )
                continue

            metrics_df.to_csv(str(metrics_file_path))

        final_map_data = metrics_df.tail(1).copy()
        final_map_data["method"] = run_dir_path.name
        cumulative_metrics_data.append(final_map_data)

    # Plot cumulative metrics
    cumulative_metrics_df = pd.concat(
        cumulative_metrics_data,
        axis=0,
        ignore_index=True,
    )

    cumulative_metrics_file_path = runs_dir_path / "metrics.csv"
    cumulative_metrics_df.to_csv(str(cumulative_metrics_file_path))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple runs against grountruth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing mapping results for the same scan.",
    )

    parser.add_argument(
        "--gt-pointcloud-file",
        required=True,
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing the groundtruth scan data.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing metrics should be recomputed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_and_compare_runs(
        gt_pointcloud_file_path=args.gt_pointcloud_file,
        runs_dir_path=args.runs_dir,
        overwrite=args.overwrite,
    )
