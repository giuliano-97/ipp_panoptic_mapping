import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_metrics(
    metrics_file_path: Path,
    title: str,
):
    assert metrics_file_path.is_file()

    metrics = pd.read_csv(metrics_file_path).sort_values("Name")
    metrics["Frame"] = pd.Series(list(range(0, len(metrics) * 50, 50)))

    sns.lineplot(
        x="Frame",
        y="value",
        hue="variable",
        data=pd.melt(metrics.drop(["Name"], axis=1), ["Frame"]),
    ).set_title(title)

    plt.grid()
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize panoptic mapping metrics in the given file."
    )

    parser.add_argument(
        "metrics_file_path",
        type=lambda p: Path(p).absolute(),
        help="Path to the metrics file.",
    )

    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default="Metrics",
        help="The title of the plot",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_metrics(
        args.metrics_file_path,
        args.title,
    )
