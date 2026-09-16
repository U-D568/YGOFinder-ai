import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras import Sequential, layers


class EmbeddingModel:
    def __init__(self, input_shape=(224, 224, 3), model_path=None, **kargs):
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            return
        self._backbone = EfficientNetV2B0(
            include_top=True,
            weights="imagenet",
            classifier_activation=None,
            input_shape=input_shape,
        )
        self._head = Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256),
            ]
        )
        self.model = Sequential([self._backbone, self._head])

    def __call__(self, inputs):
        out = self.model(inputs)
        return tf.math.l2_normalize(out, axis=-1)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def save(self, path):
        self.model.save(path)

    def make_matrix(self, dataset, batch_size=64):
        matrix = []
        for batch in dataset.batch(batch_size):
            pred = self.model.predict(batch, batch_size=batch_size, verbose=False)
            matrix.append(pred)
        matrix = tf.concat(matrix, axis=0)
        return matrix

    def predict(self, inputs):
        return self.model(inputs)
