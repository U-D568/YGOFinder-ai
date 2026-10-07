import os
import sys
import gc

import tensorflow as tf

sys.path.append(f"{os.getcwd()}")
from training.data.data_loaders.tf.embedding_dataset import EmbeddingDataset
from training.data.augmentation.tf.embeds import EmbeddingAugmentation
from structures.embedding_matrix import EmbeddingMatrix
from training.loss.tf.loss import cosine_distance
from models.embedding_model import EmbeddingModel
from training.configs.mapper.embedding import load_embedding_config


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        raise e


def main():
    config = load_embedding_config()

    # preprocessor
    augmentation = EmbeddingAugmentation(
        config.training.augmentation.min_ratio,
        config.training.augmentation.max_ratio,
    )

    # initalize hyper-parameters
    model = EmbeddingModel(config.model.input_shape)
    checkpoint_path = config.checkpoint.directory / config.checkpoint.save_best_as
    model.load(str(checkpoint_path))

    # data preparation
    train_dataset = EmbeddingDataset.load(
        str(config.data.train_csv), str(config.data.card_image_dir)
    )
    valid_dataset = EmbeddingDataset.load(
        str(config.data.valid_csv), str(config.data.card_image_dir)
    )
    valid_dataset = valid_dataset + train_dataset
    valid_matrix = EmbeddingMatrix(model, valid_dataset)
    valid_matrix.update_matrix()

    valid_dataset = EmbeddingDataset.load(
        str(config.data.valid_csv), str(config.data.card_image_dir)
    )
    gc.collect()

    hit_count = 0
    false_data = []
    false_pred = []
    for batch in valid_dataset.dataset.batch(config.training.validation_batch_size):
        anchor_img, index = batch
        positive_img = augmentation(anchor_img)

        pred_positive = model(positive_img)
        result = cosine_distance(
            valid_matrix.matrix[None, :], pred_positive[:, None, :]
        )
        pred_index = tf.argmin(result, axis=1, output_type=tf.int32)
        hit = tf.math.equal(index, pred_index)
        hit_count += tf.math.count_nonzero(hit).numpy()
        for i, h in enumerate(hit):
            if h == True:
                continue
            false_data.append(index[i])
            false_pred.append(pred_index[i])

    print(f"hit count: {hit_count} out of {len(valid_dataset)}")
    print(f"accuracy: {hit_count / len(valid_dataset) * 100}%")

if __name__ == "__main__":
    main()
