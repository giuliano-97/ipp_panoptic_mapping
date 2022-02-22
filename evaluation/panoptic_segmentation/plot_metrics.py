import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import PQ_SQ_RQ_KEYS


def plot_metrics(
    metrics_file_path: Path,
    window_size: int,
):
    assert metrics_file_path.is_file()
    output_dir_path = metrics_file_path.parent

    # Plot the metrics
    metrics_df = pd.read_csv(metrics_file_path)[["FrameID"] + PQ_SQ_RQ_KEYS]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))

    # FIXME: this should be set in the constants file (?)
    pivot_column = "FrameID"

    # Apply smoothing
    metrics_df[PQ_SQ_RQ_KEYS] = (
        metrics_df[PQ_SQ_RQ_KEYS]
        .rolling(window=window_size, center=True)
        .mean()
        .fillna(method="bfill")
    )

    # Plot all metrics in one box
    metrics_df_long = pd.melt(metrics_df, id_vars=[pivot_column], var_name="metric")
    sns.lineplot(
        data=metrics_df_long,
        x="FrameID",
        y="value",
        hue="metric",
        style="metric",
        ax=ax4,
    )

    for metric, ax in [("PQ", ax1), ("SQ", ax2), ("RQ", ax3)]:
        df = metrics_df[["FrameID", metric]].copy()

        df_long = pd.melt(df, id_vars=[pivot_column], var_name="metric")
        sns.lineplot(
            data=df_long,
            x="FrameID",
            y="value",
            hue="metric",
            style="metric",
            ax=ax,
        )

    fig.savefig(str(output_dir_path / "pq_sq_rq_trend.png"))


def _parse_args():

    parser = argparse.ArgumentParser(description="Plot panoptic segmentation metrics")

    parser.add_argument(
        "metrics_file",
        type=lambda p: Path(p).absolute(),
        help="Path to the CSV metrics file",
    )

    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=10,
        help="Window size for moving average smoothing",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_metrics(
        args.metrics_file,
        window_size=args.window_size,
    )
