from pathlib import Path

import pandas as pd
import tensorflow as tf


class EmbeddingDataset:
    @staticmethod
    def load(path, image_dir="training/datasets/card_images_small"):
        df = pd.read_csv(path)
        img_path = df["id"].tolist()
        img_path = [str(Path(image_dir) / f"{card_id}.jpg") for card_id in img_path]
        is_pendulum = list(
            map(lambda x: x.lower().startswith("pendulum"), df["type"].tolist())
        )
        return EmbeddingDataset(img_path, is_pendulum)

    def __init__(self, img_path, is_pendulum):
        self.img_path = img_path
        self.is_pendulum = is_pendulum

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

        self.dataset = self._build_dataset()

    def __len__(self):
        return len(self.img_path)

    def __add__(self, other):
        new_img_path = self.img_path + other.img_path
        new_is_pendulum = self.is_pendulum + other.is_pendulum
        return EmbeddingDataset(new_img_path, new_is_pendulum)

    def __getitem__(self, index):
        img = self._load_image(self.img_path[index])
        return self._preprocess(img, self.is_pendulum[index])

    def _generator(self):
        for path, is_pendulum in iter(zip(self.img_path, self.is_pendulum)):
            yield path, is_pendulum

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _preprocess(self, image, is_pendulum):
        crop_box = self.pendulum_pos if is_pendulum else self.normal_pos
        crop_img = tf.image.crop_and_resize(
            image=tf.expand_dims(image, axis=0),
            boxes=crop_box,
            box_indices=[0],
            crop_size=self.input_size,
            method="bilinear",
        )
        return tf.squeeze(crop_img, axis=0) / 255.0

    def _build_dataset(self):
        img_dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.bool),
            ),
        )
        img_dataset = img_dataset.map(
            lambda path, is_pendulum: self._preprocess(
                self._load_image(path), is_pendulum
            )
        )
        index = tf.data.Dataset.from_tensor_slices(list(range(self.__len__())))
        dataset = tf.data.Dataset.zip(img_dataset, index)
        return dataset
