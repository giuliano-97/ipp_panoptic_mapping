import argparse
from configparser import Interpolation
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px

from evaluation.panoptic_mapping.constants import PQ_SQ_RQ_KEYS

logging.basicConfig(level=logging.INFO)

_MAP_LOGGING_INTERVAL = 25
_FRAME_SKIP = 2


class MetricsAggregator:
    def __init__(self, pano_seg_metrics: Optional[pd.DataFrame] = None):
        self.pano_seg_metrics: pd.DataFrame = pano_seg_metrics
        self.pano_map_runs_metrics: Dict[str, pd.DataFrame] = dict()

    def add_panoptic_mapping_metrics(self, name: str, data: pd.DataFrame):
        self.pano_map_runs_metrics.update({name: data})

    def get_trend_metrics_by_run(self) -> Dict[str, pd.DataFrame]:
        res = dict()
        for name, data in self.pano_map_runs_metrics.items():
            # Convert mapping metrics indices so they match 2D frame ids
            # FIXME: these params should be read from some metadata file
            data_interpolated = data.copy()

            # Compute precision and recall
            data_interpolated["Precision"] = (
                data_interpolated["TP"]
                / (data_interpolated["TP"] + data_interpolated["FP"])
            ).fillna(0)
            data_interpolated["Recall"] = (
                data_interpolated["TP"]
                / (data_interpolated["TP"] + data_interpolated["FN"])
            ).fillna(0)

            # Drop count metrics
            data_interpolated.drop(["TP", "FP", "FN"], axis=1, inplace=True)

            data_interpolated["FrameID"] = (
                data_interpolated["FrameID"] * _FRAME_SKIP * _MAP_LOGGING_INTERVAL
            )
            # Append 2D segmentation
            if self.pano_seg_metrics is not None:
                # Get 2D panoptic quality
                pano_seg_pq = self.pano_seg_metrics[["FrameID", "PQ"]].rename(
                    columns={"PQ": "PQ_2D"}
                )
                # Smooth with rolling average
                pano_seg_pq["PQ_2D"] = (
                    pano_seg_pq["PQ_2D"]
                    .rolling(window=10, center=True)
                    .mean()
                    .fillna(method="bfill")
                )
                # Append and fill NaN rows using interpolation
                data_interpolated = (
                    pd.merge(data_interpolated, pano_seg_pq, on="FrameID", how="outer")
                    .sort_values("FrameID")
                    .interpolate()
                )

            res.update({name: data_interpolated})

        return res

    def get_trend_metrics_by_metric(self) -> pd.DataFrame:
        res = dict()
        trend_data_by_run = self.get_trend_metrics_by_run()
        for metric in PQ_SQ_RQ_KEYS + ["Precision", "Recall"]:
            # Accumulate data for this metric
            data = dict()
            for run_name, run_data in trend_data_by_run.items():
                data[run_name] = run_data.set_index("FrameID")[metric]
            res[metric] = pd.DataFrame(data).reset_index()
        return res

    def get_cumulative_metrics(self) -> pd.DataFrame:
        # For each run, get the metrics at the last frame
        cumulative_metrics = []
        for name, data in self.pano_map_runs_metrics.items():
            last_frame_data = data.tail(1).copy()
            last_frame_data["Method"] = name
            cumulative_metrics.append(last_frame_data)

        # Concatenate in one data frame
        return pd.concat(
            cumulative_metrics,
            axis=0,
            ignore_index=True,
        ).drop(["FrameID"], axis=1)


def load_metrics(runs_dir_path: Path, scan_dir_path: Path):

    # Load panoptic seg metrics
    pano_seg_metrics_file_path = scan_dir_path / "panoptic_pred" / "metrics.csv"
    if pano_seg_metrics_file_path.is_file():
        pano_seg_metrics = pd.read_csv(pano_seg_metrics_file_path)
    else:
        pano_seg_metrics = None
        logging.warning("2D panoptic segmentation metrics not found!")

    metrics_aggregator = MetricsAggregator(pano_seg_metrics=pano_seg_metrics)

    # Collect all the panoptic mapping metrics files
    for run_dir_path in [p for p in runs_dir_path.iterdir() if p.is_dir()]:
        metrics_file_path = run_dir_path / "metrics.csv"
        if not metrics_file_path.is_file():
            continue

        # Load metrics and add them to the list
        metrics_aggregator.add_panoptic_mapping_metrics(
            name=run_dir_path.name,
            data=pd.read_csv(metrics_file_path),
        )

    return metrics_aggregator


def main(
    runs_dir_path: Path,
    scan_dir_path: Path,
):
    assert runs_dir_path.is_dir()
    assert scan_dir_path.is_dir()

    metrics_aggregator = load_metrics(
        runs_dir_path=runs_dir_path,
        scan_dir_path=scan_dir_path,
    )

    # Plot trend metrics for each run
    for run_name, data in metrics_aggregator.get_trend_metrics_by_run().items():
        # Create trend plot
        fig = px.line(
            pd.melt(data, id_vars=["FrameID"], var_name="metric"),
            x="FrameID",
            y="value",
            color="metric",
        )

        # Set title and fix y axis range to [0,1]
        fig.update_layout(
            title={
                "text": run_name,
                "y": 0.94,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        fig.update_yaxes(range=[0, 1.1])  # 1.1 to leave a bit of margin above

        # Save interactive HTML plot
        interactive_plot_file_path = runs_dir_path / run_name / "trend_metrics.html"
        fig.write_html(interactive_plot_file_path)

    for metric, data in metrics_aggregator.get_trend_metrics_by_metric().items():
        # Create trend plot
        fig = px.line(
            pd.melt(data, id_vars=["FrameID"], var_name="run"),
            x="FrameID",
            y="value",
            color="run",
        )

        # Set title and fix y axis range to [0,1]
        fig.update_layout(
            title={
                "text": metric,
                "y": 0.94,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        fig.update_yaxes(range=[0, 1.1])  # 1.1 to leave a bit of margin above

        # Save interactive HTML plot
        interactive_plot_file_path = runs_dir_path / f"{metric}_trend.html"
        fig.write_html(interactive_plot_file_path)

    # Plot cumulative metrics
    cumulative_metrics = metrics_aggregator.get_cumulative_metrics()[
        PQ_SQ_RQ_KEYS + ["Method"]
    ]
    fig = px.bar(
        pd.melt(cumulative_metrics, id_vars=["Method"], var_name="metric"),
        x="Method",
        y="value",
        color="metric",
        barmode="group",
    )
    fig.update_yaxes(range=[0, 1.1])  # 1.1 to leave a bit of margin above
    interactive_plot_file_path = runs_dir_path / f"final_comparison.html"
    fig.write_html(interactive_plot_file_path)


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Plot the evaluation metrics for a collection of runs on a scan.",
    )

    parser.add_argument(
        "runs_dir_path",
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "scan_dir_path",
        type=lambda p: Path(p).absolute(),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        runs_dir_path=args.runs_dir_path,
        scan_dir_path=args.scan_dir_path,
    )
