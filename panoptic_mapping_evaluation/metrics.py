import logging
from typing import Dict, List

import numpy as np

from common import NYU40_STUFF_CLASSES, NYU40_THING_CLASSES


def _set_iou(array1: np.ndarray, array2: np.ndarray):
    intersection = np.intersect1d(array1, array2).size
    union = np.union1d(array1, array2).size
    if union == 0:
        return 0.0
    return intersection / union


def mean_iou(
    pred_voxel_segs: Dict[int, List[np.ndarray]],
    gt_voxel_segs: Dict[int, List[np.ndarray]],
):
    num_classes = len(gt_voxel_segs)
    iou_per_class = {c: 0.0 for c in gt_voxel_segs.keys()}
    for c in gt_voxel_segs.keys():
        if len(pred_voxel_segs[c]) == 0:
            continue
        # Merge list of segments into one
        pred_mask = np.concatenate(pred_voxel_segs[c])
        gt_mask = np.concatenate(gt_voxel_segs[c])
        # Compute iou
        iou_per_class[c] = _set_iou(pred_mask, gt_mask)
    # Return the average
    return sum(iou_per_class.values()) / num_classes


def panoptic_reconstruction_quality(
    pred_voxel_segs: Dict[int, List[np.ndarray]],
    gt_voxel_segs: Dict[int, List[np.ndarray]],
):
    # Match segments
    tp_per_class = {c: 0 for c in gt_voxel_segs.keys()}
    tp_iou_per_class = {c: [] for c in gt_voxel_segs.keys()}
    fp_per_class = {c: 0 for c in gt_voxel_segs.keys()}
    fn_per_class = {c: 0 for c in gt_voxel_segs.keys()}
    for c in gt_voxel_segs.keys():
        if c not in pred_voxel_segs:
            logging.warning(f"No segments were predicted for class {c}!")
            continue
        gt_matched = set()
        for pred_seg in pred_voxel_segs[c]:
            best_match = -1
            best_match_iou = 0
            for gt_seg_idx, gt_seg in enumerate(gt_voxel_segs[c]):
                if gt_seg_idx not in gt_matched:
                    iou = _set_iou(pred_seg, gt_seg)
                    if iou > 0.25 and iou > best_match_iou:
                        best_match = gt_seg_idx
                        best_match_iou = iou
            if best_match == -1:
                fp_per_class[c] += 1
            else:
                tp_per_class[c] += 1
                tp_iou_per_class[c].append(best_match_iou)
                gt_matched.add(gt_seg_idx)
        for gt_seg_idx, _ in enumerate(gt_voxel_segs[c]):
            if gt_seg_idx not in gt_matched:
                fn_per_class[c] = +1

    # Evaluate prq, srq, rrq
    srq_per_class = {c: 0.0 for c in gt_voxel_segs.keys()}
    rrq_per_class = {c: 0.0 for c in gt_voxel_segs.keys()}
    prq_per_class = {c: 0.0 for c in gt_voxel_segs.keys()}
    for c in gt_voxel_segs.keys():
        # Skip if there are no tps
        if tp_per_class[c] == 0:
            continue
        srq_per_class[c] = sum(tp_iou_per_class[c]) / tp_per_class[c]
        rrq_per_class[c] = tp_per_class[c] / (
            tp_per_class[c] + 0.5 * fp_per_class[c] + 0.5 * fn_per_class[c]
        )
        prq_per_class[c] = srq_per_class[c] * rrq_per_class[c]

    # Compute cumulative metrics
    num_classes = len(gt_voxel_segs)
    srq = sum(srq_per_class.values()) / num_classes
    rrq = sum(rrq_per_class.values()) / num_classes
    prq = sum(prq_per_class.values()) / num_classes
    tp = sum(tp_per_class.values())
    fp = sum(fp_per_class.values())
    fn = sum(fn_per_class.values())

    # Compute prq for stuff and things
    thing_classes = [i for i in NYU40_THING_CLASSES if i in gt_voxel_segs.keys()]
    stuff_classes = [i for i in NYU40_STUFF_CLASSES if i in gt_voxel_segs.keys()]
    prq_thing = sum(
        {c: v for c, v in prq_per_class.items() if c in thing_classes}.values()
    ) / len(thing_classes)
    prq_stuff = sum(
        {c: v for c, v in prq_per_class.items() if c in stuff_classes}.values()
    ) / len(stuff_classes)

    return {
        "PRQ": prq,
        "PRQ_th": prq_thing,
        "PRQ_st": prq_stuff,
        "SRQ": srq,
        "RRQ": rrq,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
