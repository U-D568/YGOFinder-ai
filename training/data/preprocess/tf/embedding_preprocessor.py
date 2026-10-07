import tensorflow as tf

class EmbeddingPreprocessor:
    def __init__(self):
        self.input_size = (224, 224)
        self.image_size = [391, 268]  # height, width
        self.normal_pos = tf.convert_to_tensor([[72, 32, 275, 236]]) / tf.tile(
            self.image_size, [2]
        )  # y1, x1, y2, x2 (normalized)
        self.normal_pos = tf.cast(self.normal_pos, dtype=tf.float32)
        self.pendulum_pos = tf.convert_to_tensor([[70, 17, 242, 250]]) / tf.tile(
            self.image_size, [2]
        )
        self.pendulum_pos = tf.cast(self.pendulum_pos, dtype=tf.float32)

    def resize(self, inputs):
        return tf.image.resize(inputs, size=self.input_size) / 255.0

    def __call__(self, inputs, is_pendulum):
        input_shape = inputs.shape
        if len(input_shape) == 3:
            inputs = tf.expand_dims(inputs, axis=0)
        crop_box = self.pendulum_pos if is_pendulum else self.normal_pos
        crop_img = tf.image.crop_and_resize(
            image=inputs,
            boxes=crop_box,
            box_indices=[0],
            crop_size=self.input_size,
            method="bilinear",
        )

        if len(input_shape) == 3:
            return tf.squeeze(crop_img, axis=0) / 255.0
        else:
            return crop_img / 255.0