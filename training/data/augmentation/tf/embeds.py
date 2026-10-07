from tensorflow.keras import layers, models

from .pixelate import RandomPixelate

class EmbeddingAugmentation:
    def __init__(self, min_ratio=0.5, max_ratio=1.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.input_shape = (224, 224)
        self.augmentation = models.Sequential(
            [
                layers.Rescaling(255),
                layers.Resizing(self.input_shape[0], self.input_shape[1]),
                RandomPixelate((self.min_ratio, self.max_ratio)),
                # layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomContrast(0.1),
                layers.RandomBrightness(0.1),
                layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
                layers.Rescaling(1.0 / 255),
            ]
        )

    def __call__(self, inputs):
        return self.augmentation(inputs)