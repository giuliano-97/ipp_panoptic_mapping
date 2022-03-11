import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.common import (
    PANOPTIC_LABEL_DIVISOR,
    NYU40_STUFF_CLASSES,
    NYU40_IGNORE_LABEL,
)

def make_detectron_stuff_label(
    segment_id,
    class_id,
    area,
):
    return {
        "id": int(segment_id),
        "isthing": False,
        "category_id": int(class_id),
        "area": int(area),
    }


def make_detectron_thing_label(
    segment_id,
    class_id,
    instance_id,
    area,
    score,
):
    return {
        "id": int(segment_id),
        "isthing": True,
        "category_id": int(class_id),
        "instance_id": int(instance_id),
        "area": int(area),
        "score": float(score),
    }


def main(pano_seg_dir_path: Path, detectron_labels_dir_path: Path):
    assert pano_seg_dir_path.is_dir()
    detectron_labels_dir_path.mkdir(exist_ok=True)

    for pano_seg_gt_file_path in pano_seg_dir_path.glob("*.png"):
        pano_seg_gt = np.array(Image.open(pano_seg_gt_file_path))

        id_image = np.zeros_like(pano_seg_gt)
        segments_info = []
        ids, areas = np.unique(pano_seg_gt, return_counts=True)
        for id, area in zip(ids, areas):
            class_id = id // PANOPTIC_LABEL_DIVISOR
            if class_id == NYU40_IGNORE_LABEL:
                continue

            if class_id in NYU40_STUFF_CLASSES:
                id_image[pano_seg_gt == id] = class_id
                segments_info.append(
                    make_detectron_stuff_label(
                        class_id,
                        class_id,
                        area,
                    )
                )
            else:
                instance_id = id % PANOPTIC_LABEL_DIVISOR
                id_image[pano_seg_gt == id] = instance_id
                segments_info.append(
                    make_detectron_thing_label(
                        instance_id, class_id, instance_id, area, 0.9
                    )
                )

        id_image_file_path = detectron_labels_dir_path / pano_seg_gt_file_path.name
        Image.fromarray(id_image).save(id_image_file_path)

        segments_info_file_path = detectron_labels_dir_path / (
            pano_seg_gt_file_path.stem + "_segments_info.json"
        )
        with segments_info_file_path.open("w") as f:
            json.dump(segments_info, f)


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Create detectron style panoptic segmentation labels."
    )

    parser.add_argument(
        "pano_seg_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to panoptic maps dir.",
    )

    parser.add_argument(
        "detectron_labels_dir",
        type=lambda p: Path(p).absolute(),
        help="Path to dir where detectron labels will be saved."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.pano_seg_dir, args.detectron_labels_dir)
