import math
import random

import tensorflow as tf

from training.loss.tf.loss import cosine_distance


class EmbeddingMatrix:
    def __init__(self, embedding_model, dataset):
        self.model = embedding_model
        self.dataset = dataset
        self.matrix = None

    def update_matrix(self, batch_size=32):
        matrix = []
        for batch in self.dataset.dataset.batch(batch_size):
            img, _ = batch
            embeds = self.model(img)
            matrix.append(embeds)
        self.matrix = tf.concat(matrix, axis=0)

    def get_hard_negative(self, anchor, anchor_index):
        taraget_index = -1
        min_distance = math.inf

        if anchor_index > 0:
            lower_matrix = self.matrix[:anchor_index]
            lower_matrix = cosine_distance(lower_matrix, anchor)
            min_index = tf.argmin(lower_matrix, axis=0, output_type=tf.int32)
            distance = lower_matrix[min_index]
            if distance < min_distance:
                min_distance = distance
                taraget_index = min_index

        if anchor_index < self.matrix.shape[0] - 1:
            upper_matrix = self.matrix[anchor_index + 1 :]
            upper_matrix = cosine_distance(upper_matrix, anchor)
            min_index = tf.argmin(upper_matrix, axis=0, output_type=tf.int32)
            distance = upper_matrix[min_index]
            if distance < min_distance:
                min_distance = distance
                taraget_index = min_index

        return taraget_index

    def get_random_negative(self, anchor_index):
        target_index = -1
        while True:
            target_index = random.randrange(len(self.dataset))
            if target_index != anchor_index:
                break
        return target_index
