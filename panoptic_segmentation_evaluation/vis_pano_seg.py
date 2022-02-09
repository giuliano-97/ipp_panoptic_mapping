import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from PIL import Image

import panoptic_segmentation_evaluation.visualization as vis_utils
from common import (
    NYU40_THING_CLASSES,
    NYU40_CLASS_IDS_TO_NAMES,
    NYU40_COLOR_PALETTE,
    PANOPTIC_LABEL_DIVISOR,
)

_COLOR_PERTURBATION = 60  # Amount of color perturbation


def colorize_pano_seg(
    pano_seg: np.ndarray,
    color_palette: np.ndarray,
):

    colorized_pano_seg = np.zeros(
        shape=(pano_seg.shape[0], pano_seg.shape[1], 3),
        dtype=np.uint8,
    )

    semantic_result = pano_seg[:, :, 0]
    instance_result = pano_seg[:, :, 1]
    unique_semantic_values = np.unique(pano_seg[:, :, 0])

    used_colors = collections.defaultdict(set)
    np_state = np.random.RandomState(0)

    for semantic_value in unique_semantic_values:
        semantic_mask = semantic_result == semantic_value
        if semantic_value in NYU40_THING_CLASSES:
            # For `thing` class, we will add a small amount of random noise to its
            # correspondingly predefined semantic segmentation colormap.
            unique_instance_values = np.unique(instance_result[semantic_mask])
            for instance_value in unique_instance_values:
                instance_mask = np.logical_and(
                    semantic_mask, instance_result == instance_value
                )

                random_color = vis_utils.perturb_color(
                    color_palette[semantic_value],
                    _COLOR_PERTURBATION,
                    used_colors[semantic_value],
                    random_state=np_state,
                )
                colorized_pano_seg[instance_mask] = random_color
        else:
            # For `stuff` class, we use the defined semantic color.
            colorized_pano_seg[semantic_mask] = color_palette[semantic_value]
            used_colors[semantic_value].add(tuple(color_palette[semantic_value]))

    return colorized_pano_seg, used_colors


def decode_single_channel_pano_seg(single_channel_pano_seg: np.ndarray):
    height, width = single_channel_pano_seg.shape[:2]
    pano_seg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    pano_seg[:, :, 0] = single_channel_pano_seg // PANOPTIC_LABEL_DIVISOR
    pano_seg[:, :, 1] = single_channel_pano_seg % PANOPTIC_LABEL_DIVISOR
    return pano_seg


def vis_pano_seg(
    raw_image_file: str,
    pano_seg_file: str,
    semantic_uncertainty_map_file: str = None,
):
    pano_seg = np.array(Image.open(pano_seg_file))
    raw_image = np.array(Image.open(raw_image_file))

    # Decode single channel panoptic
    if pano_seg.ndim == 2:
        pano_seg = decode_single_channel_pano_seg(pano_seg)
    assert np.max(pano_seg[:, :, 0] < 41), "Only nyu40 labels supported."

    color_palette = np.array(NYU40_COLOR_PALETTE)
    colorized_pano_seg, _ = colorize_pano_seg(
        pano_seg,
        color_palette,
    )

    num_plots = 3 if semantic_uncertainty_map_file is not None else 2

    plt.figure("Visualize panoptic segmentation")
    plt.subplot(1, num_plots, 1)
    plt.imshow(colorized_pano_seg)

    legend_handles = []
    for semantic_id in np.unique(pano_seg[:, :, 0]):
        if semantic_id > 0:
            legend_handles.append(
                mpatches.Patch(
                    color=color_palette[semantic_id] / 255,
                    label=NYU40_CLASS_IDS_TO_NAMES[semantic_id],
                )
            )

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1, 0),
        loc="best",
    )

    plt.subplot(1, num_plots, 2)
    plt.imshow(raw_image)
    if semantic_uncertainty_map_file is not None:
        semantic_uncertainty_map = np.array(Image.open(semantic_uncertainty_map_file))
        plt.subplot(1, num_plots, 3)
        # Display uncertainty as gradient using blue to red
        plt.imshow(semantic_uncertainty_map, cmap=cm.get_cmap("inferno"))
    plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Colorize and display panoptic segmentation result."
    )

    parser.add_argument(
        "raw_image_file",
        type=str,
        help="Path to the raw image",
    )

    parser.add_argument(
        "pano_seg_file",
        type=str,
        help="Path to the panoptic segmentation file.",
    )

    parser.add_argument(
        "semantic_uncertainty_map_file",
        type=str,
        nargs="?",
        help="Path to uncertainty map",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    vis_pano_seg(
        args.raw_image_file,
        args.pano_seg_file,
        args.semantic_uncertainty_map_file,
    )
