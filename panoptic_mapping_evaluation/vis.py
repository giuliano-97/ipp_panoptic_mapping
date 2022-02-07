from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Turn off interactive mode
plt.ioff()


def create_grouped_barplot(
    metrics_df: pd.DataFrame,
    barplot_file_path: Path,
    pivot_column: str = "method",
):
    # Convert metrics dataframe from wide to long format
    metrics_df_long = pd.melt(metrics_df, id_vars=[pivot_column], var_name="metric")
    # Create catplot
    grid = sns.catplot(
        data=metrics_df_long,
        kind="bar",
        x=pivot_column,
        y="value",
        hue="metric",
    )

    # Configure plot
    grid.legend.set_title("Metrics")

    # Save the plot
    grid.figure.savefig(str(barplot_file_path))