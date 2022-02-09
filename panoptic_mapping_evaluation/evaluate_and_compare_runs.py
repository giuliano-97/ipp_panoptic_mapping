import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from constants import MIOU_KEY, PRQ_SRQ_RRQ_KEYS, TP_FP_FN_KEYS
from graphing import save_grouped_barplot
from panoptic_mapping_evaluation.evaluate_run import evaluate_run


def _find_runs(runs_dir_path: Path) -> List[Path]:
    run_dirs = []
    for p in runs_dir_path.iterdir():
        if not p.is_dir():
            continue
        if len(list(p.glob("*.pointcloud.ply"))) > 0:
            run_dirs.append(p)
    return run_dirs


def evaluate_and_compare_runs(
    scan_dir_path: Path,
    runs_dir_path: Path,
):
    assert scan_dir_path.is_dir()

    gt_pointcloud_file_path = next(scan_dir_path.glob("*.pointcloud.ply"), None)
    if gt_pointcloud_file_path is None:
        logging.warning("Groundtruth scan data dir has no panoptic labeled pointcloud!")
        exit(1)

    run_dirs = _find_runs(runs_dir_path)
    if len(run_dirs) == 0:
        logging.warning(f"No runs to evaluate were found in {runs_dir_path.name}")

    cumulative_metrics_data = []

    for run_dir_path in run_dirs:
        logging.info(f"Evaluating run: {run_dir_path.name}")

        metrics_file_path = run_dir_path / "metrics.csv"
        if metrics_file_path.is_file():
            metrics_df = pd.read_csv(str(metrics_file_path))
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

    prq_srq_rrq_plot_file_path = runs_dir_path / "prq_srq_rrq.png"
    save_grouped_barplot(
        cumulative_metrics_df[["method"] + PRQ_SRQ_RRQ_KEYS],
        prq_srq_rrq_plot_file_path,
    )

    tp_fp_fn_plot_file_path = runs_dir_path / "tp_fp_fn.png"
    save_grouped_barplot(
        cumulative_metrics_df[["method"] + TP_FP_FN_KEYS],
        tp_fp_fn_plot_file_path,
    )

    miou_plot_file_path = runs_dir_path / "miou.png"
    save_grouped_barplot(
        cumulative_metrics_df[["method", MIOU_KEY]],
        miou_plot_file_path,
    )


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
        "--scan-dir",
        required=True,
        type=lambda p: Path(p).absolute(),
        help="Path to the directory containing the groundtruth scan data.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_and_compare_runs(
        scan_dir_path=args.scan_dir,
        runs_dir_path=args.runs_dir,
    )
