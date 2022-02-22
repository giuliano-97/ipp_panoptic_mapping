import argparse
from pathlib import Path

import pandas as pd

from utils.graphing import save_trend_lineplot


def plot_metrics(
    metrics_file_path: Path,
):
    assert metrics_file_path.is_file()
    output_dir_path = metrics_file_path.parent
    # Plot the metrics
    metrics_df = pd.read_csv(metrics_file_path).fillna(0)
    for metric in ["PQ", "SQ", "RQ"]:
        trend_plot_file_path = output_dir_path / f"{metric.lower()}_trend.png"
        data = metrics_df[["FrameID", metric]].copy()

        # Add smoothed version
        data[metric + "_mov_avg"] = (
            data[metric].rolling(window=10, center=False).mean().fillna(method="bfill")
        )

        save_trend_lineplot(
            data,
            trend_plot_file_path,
        )


def _parse_args():

    parser = argparse.ArgumentParser(description="Plot panoptic segmentation metrics")

    parser.add_argument(
        "metrics_file",
        type=lambda p: Path(p).absolute(),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_metrics(args.metrics_file)
