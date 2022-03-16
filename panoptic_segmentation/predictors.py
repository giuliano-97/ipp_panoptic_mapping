from abc import ABC, abstractmethod
from tkinter import Pack
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.common import (
    NYU40_IGNORE_LABEL,
    NYU40_THING_CLASSES,
    NYU40_STUFF_CLASSES,
    PANOPTIC_LABEL_DIVISOR,
)


class Predictor(ABC):
    @abstractmethod
    def predict(
        self, image: np.ndarray
    ) -> Dict[str,]:
        pass


class MaXDeepLab(Predictor):

    PIXEL_SPACE_MASK_LOGITS_KEY = "pixel_space_mask_logits"
    PANOPTIC_PRED_KEY = "panoptic_pred"
    TRANSFOMER_CLASS_LOGITS_KEY = "transformer_class_logits"
    SEMANTIC_LOGITS_KEY = "semantic_logits"
    TRANSFOMER_CLASS_CONFIDENCE_THRESH = 0.7
    MASK_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _get_detected_masks(
        class_logits: tf.Tensor,
    ) -> Dict[str, np.ndarray]:
        # Note: the last class represents the null label and we don't consider it
        # when extracting the detected segments, so don't include the probability
        class_probs = tf.nn.softmax(class_logits, axis=-1)[..., :-1]
        masks_class_confidence = tf.reduce_max(
            class_probs,
            axis=-1,
            keepdims=False,
        )

        # Filter masks with class confidence less than the threshold.
        thresholded_mask = tf.cast(
            tf.greater_equal(
                masks_class_confidence,
                MaXDeepLab.TRANSFOMER_CLASS_CONFIDENCE_THRESH,
            ),
            tf.float32,
        )

        detected_mask_indices = tf.where(tf.greater(thresholded_mask, 0.5))[:, 0]
        detected_mask_class_logits = tf.gather(
            class_logits,
            detected_mask_indices,
            axis=0,
        )

        detected_mask_class_pred = tf.argmax(detected_mask_class_logits, axis=-1)

        return (
            detected_mask_indices,
            detected_mask_class_logits,
            detected_mask_class_pred,
        )

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        res_dict = self.model(tf.cast(image, tf.uint8))

        class_logits = tf.squeeze(
            res_dict[MaXDeepLab.TRANSFOMER_CLASS_LOGITS_KEY], axis=0
        )

        # Filter masks by class score
        detected_mask_indices, _, _ = MaXDeepLab._get_detected_masks(class_logits)

        # Get mask logits and resize to native resolution
        # Note: maybe tf.image.resize does the same thing
        mask_logits = tf.squeeze(
            tf.compat.v1.image.resize_bilinear(
                res_dict[self.PIXEL_SPACE_MASK_LOGITS_KEY],
                (image.shape[0] + 1, image.shape[1] + 1),  # +1 because of the cropping
                align_corners=True,
            ),
            axis=0,
        )

        # Drop the last row and column to match the image shape
        mask_logits = mask_logits[:-1, :-1]

        # Get the pixel mask logits for the detected segments
        detected_mask_logits = tf.gather(
            mask_logits,
            detected_mask_indices,
            axis=-1,
        )

        # Get post-processed panoptic prediction
        panoptic_pred = tf.squeeze(res_dict[MaXDeepLab.PANOPTIC_PRED_KEY], axis=0)

        # Create segment infos
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

        semantic_logits = tf.squeeze(res_dict[MaXDeepLab.SEMANTIC_LOGITS_KEY], axis=0)

        return {
            "panoptic_pred": panoptic_pred.numpy(),
            "mask_logits": detected_mask_logits.numpy(),
            "semantic_logits": semantic_logits.numpy(),
        }
