import collections
import logging

import numpy as np

from utils.common import (
    NYU40_IGNORE_LABEL,
    NYU40_STUFF_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
    NYU40_THING_CLASSES,
)


def perturb_color(
    color,
    noise,
    used_colors=None,
    max_trials=50,
    random_state=None,
):
    """Pertrubs the color with some noise.

    If `used_colors` is not None, we will return the color that has
    not appeared before in it.

    Args:
      color: A numpy array with three elements [R, G, B].
      noise: Integer, specifying the amount of perturbing noise.
      used_colors: A set, used to keep track of used colors.
      max_trials: An integer, maximum trials to generate random color.
      random_state: An optional np.random.RandomState. If passed, will be used to
        generate random numbers.

    Returns:
      A perturbed color that has not appeared in used_colors.
    """
    for _ in range(max_trials):
        if random_state is not None:
            random_color = color + random_state.randint(
                low=-noise, high=noise + 1, size=3
            )
        else:
            random_color = color + np.random.randint(low=-noise, high=noise + 1, size=3)
        random_color = np.maximum(0, np.minimum(255, random_color))
        if used_colors is None:
            return random_color
        elif tuple(random_color) not in used_colors:
            used_colors.add(tuple(random_color))
            return random_color
    logging.warning("Using duplicate random color.")
    return random_color


def colorize_panoptic_labels(panoptic_labels, color_palette):

    if len(panoptic_labels.shape) == 2:
        colors = np.zeros(
            shape=(panoptic_labels.shape[0], panoptic_labels.shape[1], 3),
            dtype=np.uint8,
        )
    else:
        colors = np.zeros(shape=(panoptic_labels.shape[0], 3), dtype=np.uint8)

    semantic_labels = panoptic_labels // PANOPTIC_LABEL_DIVISOR
    instance_ids = panoptic_labels % PANOPTIC_LABEL_DIVISOR
    unique_semantic_values = np.unique(semantic_labels)

    used_colors = collections.defaultdict(set)
    np_state = np.random.RandomState(0)

    for semantic_value in unique_semantic_values:
        semantic_mask = semantic_labels == semantic_value
        if (
            semantic_value == NYU40_IGNORE_LABEL
            or semantic_value in NYU40_STUFF_CLASSES
        ):
            # For `stuff` class, we use the defined semantic color.
            colors[semantic_mask] = color_palette[semantic_value]
            used_colors[semantic_value].add(tuple(color_palette[semantic_value]))
        elif semantic_value in NYU40_THING_CLASSES:
            # For `thing` class, we will add a small amount of random noise to its
            # correspondingly predefined semantic segmentation colormap.
            unique_instance_values = np.unique(instance_ids[semantic_mask])
            for instance_value in unique_instance_values:
                instance_mask = np.logical_and(
                    semantic_mask, instance_ids == instance_value
                )

                random_color = perturb_color(
                    color_palette[semantic_value],
                    60,
                    used_colors[semantic_value],
                    random_state=np_state,
                )
                colors[instance_mask] = random_color

    return colors, used_colors
