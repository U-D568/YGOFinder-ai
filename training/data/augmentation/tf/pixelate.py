from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


class RandomPixelate(layers.Layer):
    def __init__(self, factor: Tuple[float, float], **kwarg):
        super().__init__(**kwarg)
        self.minval = min(factor)
        self.maxval = max(factor)

    def call(self, x: tf.Tensor):
        assert len(x.shape) == 3 or len(x.shape) == 4, "tensor's dim have to 3 or 4"
        # assign height and width depends on tensor shape
        shape = x.shape[:2] if len(x.shape) == 3 else x.shape[1:3]
        zoom_ratio = tf.random.uniform(
            [], minval=self.minval, maxval=self.maxval, dtype=tf.float32
        )
        new_shape = tf.math.round(tf.cast(shape, tf.float32) * zoom_ratio)
        new_shape = tf.cast(new_shape, tf.int32)
        image = tf.image.resize(x, new_shape)
        image = tf.image.resize(image, shape)

        return image

