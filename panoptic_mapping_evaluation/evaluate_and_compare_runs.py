import argparse
import logging
from pathlib import Path

import pandas as pd


from constants import MIOU_KEY, PRQ_SRQ_RRQ_KEYS, TP_FP_FN_KEYS
from vis import create_grouped_barplot
from evaluate_run import evaluate_run


def _find_runs(runs_dir_path: Path):
    run_dirs = []
    for p in runs_dir_path.iterdir():
        if not p.is_dir():
            continue
        if len(list(p.glob("*.voxel_segs.json"))) > 0:
            run_dirs.append(p)
    return run_dirs


def evaluate_and_compare_runs(
    scan_dir_path: Path,
    runs_dir_path: Path,
):
    assert scan_dir_path.is_dir()
    gt_voxel_segs_file_path = next(scan_dir_path.glob("*.voxel_segs.json"), None)
    if gt_voxel_segs_file_path is None:
        logging.warning("Groundtruth scan data dir has no voxel segs.json file.")
        exit(1)

    run_dirs = _find_runs(runs_dir_path)
    if len(run_dirs) == 0:
        logging.warning(f"No runs to evaluate were found in {runs_dir_path.name}")

    cumulative_metrics_data = []

    for run_dir_path in run_dirs:
        # TODO: hack to determine whether ids should be remapped
        # better to do this beforehand or make this step unnecessary
        map_ids_to_panoptic = "single_tsdf" in run_dir_path.name
        metrics_df = evaluate_run(
            run_dir_path=run_dir_path,
            gt_voxel_segs_file_path=gt_voxel_segs_file_path,
            map_ids_to_panoptic=map_ids_to_panoptic,
        )

        if metrics_df is None:
            logging.warning(f"Run {run_dir_path.name} could not be evaluated. Skipped.")
            continue

        final_map_data = metrics_df.tail(1).copy()
        final_map_data["method"] = run_dir_path.name
        cumulative_metrics_data.append(final_map_data)

    # Plot cumulative metrics
    cumulative_metrics_df = pd.concat(
        cumulative_metrics_data,
        axis=0,
        ignore_index=True,
    )

    prq_srq_rrq_plot_file_path = runs_dir_path / "prq_srq_rrq.png"
    create_grouped_barplot(
        cumulative_metrics_df[["method"] + PRQ_SRQ_RRQ_KEYS],
        prq_srq_rrq_plot_file_path,
    )

    tp_fp_fn_plot_file_path = runs_dir_path / "tp_fp_fn.png"
    create_grouped_barplot(
        cumulative_metrics_df[["method"] + TP_FP_FN_KEYS],
        tp_fp_fn_plot_file_path,
    )

    miou_plot_file_path = runs_dir_path / "miou.png"
    create_grouped_barplot(
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
