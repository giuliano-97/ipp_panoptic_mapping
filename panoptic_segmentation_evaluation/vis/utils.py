import logging

import numpy as np

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
