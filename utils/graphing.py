from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Turn off interactive mode
plt.ioff()


_DEFAULT_FIGURE_SIZE=(16, 9)


def save_grouped_barplot(
    metrics_df: pd.DataFrame,
    barplot_file_path: Path,
    pivot_column: str = "method",
):
    plt.figure(figsize=(_DEFAULT_FIGURE_SIZE))

    # Convert metrics dataframe from wide to long format
    metrics_df_long = pd.melt(metrics_df, id_vars=[pivot_column], var_name="metric")
    # Create catplot
    grid = sns.catplot(
        data=metrics_df_long,
        kind="bar",
        x=pivot_column,
        y="value",
        hue="metric",
        height=_DEFAULT_FIGURE_SIZE[1],
        aspect=_DEFAULT_FIGURE_SIZE[0] / _DEFAULT_FIGURE_SIZE[1],
    )

    # Configure plot
    grid.legend.set_title("Metrics")

    # Save the plot
    grid.figure.savefig(str(barplot_file_path))

    # Clean up
    grid.figure.clear()
    plt.close(grid.figure)


def save_trend_lineplot(
    metrics_df: pd.DataFrame,
    lineplot_file_path: Path,
    pivot_column: Optional[str] = "FrameID",
):
    plt.figure(figsize=_DEFAULT_FIGURE_SIZE)
    metrics_df_long = pd.melt(metrics_df, id_vars=[pivot_column], var_name="metric")
    ax = sns.lineplot(
        data=metrics_df_long, x=pivot_column, y="value", hue="metric", style="metric"
    )

    ax.figure.savefig(str(lineplot_file_path))

    # Clean up
    ax.figure.clear()
    plt.close(ax.figure)
