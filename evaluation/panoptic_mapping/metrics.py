from operator import gt
from typing import Mapping, Dict

import numpy as np

from constants import (
    PQ_KEY,
    PQ_THING_KEY,
    PQ_STUFF_KEY,
    SQ_KEY,
    RQ_KEY,
    TP_KEY,
    FN_KEY,
    FP_KEY,
    MIOU_KEY,
)
from segment_matching import match_segments
from utils.common import (
    NYU40_THING_CLASSES,
    NYU40_STUFF_CLASSES,
    SCANNET_NYU40_EVALUATION_CLASSES,
    NYU40_NUM_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
)


_THING_CLASSES_MASK = np.isin(np.arange(NYU40_NUM_CLASSES), NYU40_THING_CLASSES)
_STUFF_CLASSES_MASK = np.isin(np.arange(NYU40_NUM_CLASSES), NYU40_STUFF_CLASSES)
_EVAL_CLASSES_MASK = np.isin(
    np.arange(NYU40_NUM_CLASSES), SCANNET_NYU40_EVALUATION_CLASSES
)


def mean_iou(gt_labels: np.ndarray, pred_labels: np.ndarray):
    iou_per_class = np.zeros(NYU40_NUM_CLASSES, dtype=np.float32)

    # We evaluate IoU only over class labels
    gt_semantic_labels = gt_labels // PANOPTIC_LABEL_DIVISOR
    pred_semantic_labels = pred_labels // PANOPTIC_LABEL_DIVISOR

    # Only evaluate over classes that appear in the GT, are not the ignore label
    # and are in the list of classes on which one should evaluate
    valid_classes = np.unique(gt_semantic_labels)
    valid_classes = valid_classes[
        np.isin(
            valid_classes,
            SCANNET_NYU40_EVALUATION_CLASSES + [NYU40_NUM_CLASSES],
        )
    ]

    if valid_classes.size == 0:
        return {MIOU_KEY: 0}

    # Compute IoU for every valid class
    for class_id in valid_classes:
        gt_class_mask = gt_semantic_labels == class_id
        pred_class_mask = pred_semantic_labels == class_id

        intersection_area = np.count_nonzero(gt_class_mask & pred_class_mask)
        union_area = np.count_nonzero(gt_class_mask | pred_class_mask)

        iou_per_class[class_id] = intersection_area / union_area

    # Compute mean iou
    miou = np.mean(np.take(iou_per_class, valid_classes))

    return {MIOU_KEY: miou}


def panoptic_quality(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
):
    if gt_labels.shape != pred_labels.shape:
        raise ValueError("Label arrays must have the same shape!")

    matching_result = match_segments(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
    )

    iou_per_class = matching_result.iou_per_class
    tp_per_class = matching_result.tp_per_class
    fp_per_class = matching_result.fp_per_class
    fn_per_class = matching_result.fn_per_class

    with np.errstate(divide="ignore", invalid="ignore"):
        sq_per_class = np.nan_to_num(iou_per_class / tp_per_class)
        rq_per_class = np.nan_to_num(
            tp_per_class / (tp_per_class + 0.5 * fp_per_class + 0.5 * fn_per_class)
        )
    pq_per_class = np.multiply(sq_per_class, rq_per_class)

    # Evaluate only on classes which appear at least once in the groundtruth
    # and are in the validation classes used by the ScanNet benchmark
    valid_classes_mask = _EVAL_CLASSES_MASK & np.not_equal(
        tp_per_class + fp_per_class + fn_per_class, 0
    )

    # Eval metrics,
    qualities_per_class = np.row_stack((pq_per_class, sq_per_class, rq_per_class))
    counts_per_class = np.row_stack((tp_per_class, fp_per_class, fn_per_class))
    tp, fp, fn = np.sum(
        counts_per_class[:, valid_classes_mask],
        axis=1,
    )

    if np.count_nonzero(valid_classes_mask) > 0:
        pq, sq, rq = np.mean(
            qualities_per_class[:, valid_classes_mask],
            axis=1,
        )
    else:
        pq, sq, rq = 0, 0, 0

    # Also compute pq for thing and stuff classes only
    valid_thing_classes_mask = valid_classes_mask & _THING_CLASSES_MASK
    if np.count_nonzero(valid_thing_classes_mask) > 0:
        pq_th = np.mean(qualities_per_class[0][valid_thing_classes_mask])
    else:
        pq_th = 0

    valid_stuff_classes_mask = valid_classes_mask & _STUFF_CLASSES_MASK
    if np.count_nonzero(valid_stuff_classes_mask) > 0:
        pq_st = np.mean(qualities_per_class[0][valid_stuff_classes_mask])
    else:
        pq_st = 0

    return {
        PQ_KEY: pq,
        PQ_THING_KEY: pq_th,
        PQ_STUFF_KEY: pq_st,
        SQ_KEY: sq,
        RQ_KEY: rq,
        TP_KEY: tp,
        FP_KEY: fp,
        FN_KEY: fn,
    }
