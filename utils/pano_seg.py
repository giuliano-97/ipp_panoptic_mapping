from typing import List, Dict

import numpy as np

from evaluation.panoptic_mapping.segment_matching import match_segments
from utils.common import (
    NYU40_THING_CLASSES,
    NYU40_STUFF_CLASSES,
    NYU40_IGNORE_LABEL,
    PANOPTIC_LABEL_DIVISOR,
)


def match_and_remap_panoptic_labels(
    src_pano_seg: np.ndarray,
    dst_pano_seg: np.ndarray,
    ignore_unmatched: bool = False,
    match_threshold: float = 0.25,
) -> np.ndarray:
    assert src_pano_seg.shape == dst_pano_seg.shape

    # Match panoptic segments in the two images
    res = match_segments(
        src_pano_seg,
        dst_pano_seg,
        match_iou_threshold=match_threshold,
        fp_min_area=0,
    )

    # Copy pano seg map to remap
    used_ids = set()
    remapped_dst_pano_seg = np.zeros_like(dst_pano_seg)
    for old_id, new_id in res.tp_matches.items():
        remapped_dst_pano_seg[dst_pano_seg == old_id] = new_id
        used_ids.add(new_id)

    # Now add segments which were not matched making sure there
    # are no collisions with the remapped ids
    for unmatched_id in res.fps:
        mask = dst_pano_seg == unmatched_id
        if ignore_unmatched:
            remapped_dst_pano_seg[mask] = 0
        elif unmatched_id not in used_ids:
            remapped_dst_pano_seg[mask] = unmatched_id
            used_ids.add(unmatched_id)
        else:
            new_id = unmatched_id + 1
            while new_id in used_ids:
                new_id += 1
            remapped_dst_pano_seg[mask] = new_id
            used_ids.add(new_id)

    return remapped_dst_pano_seg


def create_segments_info(panoptic_pred: np.ndarray) -> List[Dict]:
    segment_ids, areas = np.unique(panoptic_pred, return_counts=True)
    segments_info = []
    for segment_id, area in zip(segment_ids, areas):
        if segment_id == NYU40_IGNORE_LABEL:
            continue
        class_id = segment_id // PANOPTIC_LABEL_DIVISOR

        info = {
            "id": int(segment_id),
            "category_id": int(class_id),
            "area": int(area),
        }

        if class_id in NYU40_STUFF_CLASSES:
            info["isthing"] = False
        elif class_id in NYU40_THING_CLASSES:
            instance_id = segment_id % PANOPTIC_LABEL_DIVISOR
            info["isthing"] = True
            info["instance_id"] = instance_id

        segments_info.append(info)

    return segments_info