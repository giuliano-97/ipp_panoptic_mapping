import logging

import numpy as np

import pointcloud as pcd_utils
import constants as constants
from metrics import panoptic_quality, mean_iou
from utils.common import (
    NYU40_IGNORE_LABEL,
)


logging.basicConfig(level=logging.INFO)


class PanopticPointcloudEvaluator:
    def __init__(self, gt_points, gt_labels):
        self.gt_points = gt_points
        self.gt_labels = gt_labels

        # Compute world to grid transform
        self.w2g_transform = pcd_utils.get_world_to_grid_transform(
            self.gt_points,
            voxel_size=constants.VOXEL_SIZE,
        )

        # Pre-compute groundtruth label grid
        self.gt_points_gc = pcd_utils.convert_points_to_grid_coordinates(
            self.gt_points, self.w2g_transform, max_grid_coord=None
        )

        # Pre-compute max grid coord
        self.max_grid_coord = (
            np.ceil(np.max(self.gt_points_gc, axis=0)).astype(np.int32) + 1
        )

    def evaluate(self, pred_points, pred_labels, coverage_points=None):
        # Convert predicted pointcloud points to grid coordinates
        pred_points_gc = pcd_utils.convert_points_to_grid_coordinates(
            pred_points,
            self.w2g_transform,
            self.max_grid_coord,
        )

        # Create grid with predicted labels
        pred_label_grid = pcd_utils.make_panoptic_grid(
            pred_points_gc, pred_labels, self.max_grid_coord
        )

        # Look up the label for each gt point in grid
        pred_gt_labels = pred_label_grid[tuple(self.gt_points_gc.T)]

        # If the coverage pointcloud is provided, mask the gt labels which have not been covered
        if coverage_points is not None:
            coverage_points_gc = pcd_utils.convert_points_to_grid_coordinates(
                coverage_points,
                self.w2g_transform,
                self.max_grid_coord,
            )
            coverage_grid = pcd_utils.make_occupancy_grid(
                coverage_points_gc, self.max_grid_coord
            )
            covered_gt_labels_mask = coverage_grid[tuple(self.gt_points_gc.T)]
            covered_gt_labels = np.where(
                covered_gt_labels_mask, self.gt_labels, NYU40_IGNORE_LABEL
            )
        else:
            covered_gt_labels = self.gt_labels

        # Compute metrics
        metrics_dict = dict()
        metrics_dict.update(
            panoptic_quality(
                pred_labels=pred_gt_labels,
                gt_labels=covered_gt_labels,
            )
        )
        metrics_dict.update(
            mean_iou(
                pred_labels=pred_gt_labels,
                gt_labels=covered_gt_labels,
            )
        )

        return metrics_dict
