from collections.abc import Sequence
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.image_utils import (
    random_pixelate,
    random_zoom_transition,
    make_square_shape,
    make_deck_image,
)
from utils import common


class DecklistDataset(Dataset):
    @staticmethod
    def load_from_csv(df_path, deck_size, image_dir="datasets/card_images_small/"):
        X_train = pd.read_csv(df_path)
        id_list = X_train["id"].tolist()
        id_list = list(map(lambda x: image_dir + str(x) + ".jpg", id_list))
        card_type = list(
            map(lambda x: x.lower().startswith("pendulum"), X_train["type"].tolist())
        )
        return DecklistDataset(id_list, card_type, deck_shape=deck_size)

    def __init__(
        self,
        image_pathes,
        is_pendulum,
        pixelate_scale=(0.5, 1.0),
        zoom_scale=(0.5, 1.0),
        deck_shape=60,
        num_workers=4,
    ):
        self.image_loader = common.ImageLoader(num_workers=num_workers)
        self.device = torch.device
        self.card_data = list(zip(image_pathes, is_pendulum))
        self.deck_shape = deck_shape
        self.deck_data = self.make_deck_images()
        self.pixelate_scale = pixelate_scale
        self.zoom_scale = zoom_scale

        # relative coordinates of ilust in image
        self.card_size = np.array([268, 391])
        self.normal_pos = np.array([32, 72, 236, 275]) / np.tile(self.card_size, 2)
        self.pendulum_pos = np.array([17, 70, 250, 242]) / np.tile(self.card_size, 2)

    def make_deck_images(self):
        deck_sizes = self.make_deck_sizes()
        i = 0
        deck_images = []
        random.shuffle(self.card_data)
        for size in deck_sizes:
            deck_images.append(self.card_data[i : i + size])
            i = min(i + size, len(self.card_data) - 1)

        return deck_images

    def make_deck_sizes(self):
        if isinstance(self.deck_shape, Sequence) and all(n for n in self.deck_shape):
            deck_sizes = self.make_deck_size_in_range(self.deck_shape)
        elif isinstance(self.deck_shape, int):
            deck_sizes = self.make_fixed_deck_size(self.deck_shape)
        else:
            raise ValueError(
                f"Invalid deck_shape value. It should be int or sequence of int. used value: {self.deck_shape}"
            )

        random.shuffle(deck_sizes)
        return deck_sizes

    def make_deck_size_in_range(self, deck_size_range):
        data_length = len(self.card_data)
        deck_size_list = []

        while data_length > 0:
            num = random.randint(deck_size_range[0], deck_size_range[1])
            if data_length >= num:
                deck_size_list.append(num)
                data_length -= num
            else:
                deck_size_list.append(data_length)
                data_length = 0

        random.shuffle(deck_size_list)
        return deck_size_list

    def make_fixed_deck_size(self, deck_size):
        data_length = len(self.card_data)
        div = data_length // deck_size
        mod = data_length % deck_size

        deck_size_list = [deck_size] * div
        if mod > 0:
            deck_size_list.append(mod)

        random.shuffle(deck_size_list)
        return deck_size_list

    def __len__(self):
        return len(self.deck_data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        image_pathes = [data[0] for data in self.deck_data[index]]
        card_types = dict(self.deck_data[index])

        self.image_loader.set_queue(image_pathes)
        images = self.image_loader.run()
        ids = self.image_loader.get_file_names()
        # Parallel image reads can complete in a different order from the queue.
        is_pendulum = [card_types[path] for path in ids]
        ids = list(map(common.get_filename, ids))

        xyxy = []
        for i, image in enumerate(images):
            # augment indivisual card image
            image = random_pixelate(
                image, min(self.pixelate_scale), max(self.pixelate_scale)
            )
            image, ratio, offset = random_zoom_transition(
                image, min(self.zoom_scale), max(self.zoom_scale)
            )
            images[i] = image
            h, w, _ = image.shape

            # make detection label
            pos = self.pendulum_pos.copy() if is_pendulum[i] else self.normal_pos.copy()
            pos = pos * ratio * np.array([w, h, w, h])
            pos += np.tile(offset, 2)
            pos = np.round(pos)
            xyxy.append(pos)
        xyxy = np.stack(xyxy)

        origianl_image, offset = make_deck_image(images)
        xyxy += np.tile(offset, 2)

        deck_image, ratio, offset = make_square_shape(origianl_image, 640)
        height, width, _ = deck_image.shape
        # card image position(xyxy)
        xyxy = xyxy * ratio + np.tile(offset, 2)
        xyxy[:, [0, 2]] /= width
        xyxy[:, [1, 3]] /= height

        # xyxy to xywh
        # shape -> (60, 4)
        width = xyxy[:, 2] - xyxy[:, 0]
        height = xyxy[:, 3] - xyxy[:, 1]
        xywh = xyxy.copy()
        xywh[:, 0] += width / 2
        xywh[:, 1] += height / 2
        xywh[:, 2] = width
        xywh[:, 3] = height

        return {
            "image": deck_image,
            "ids": ids,
            "xyxy": xyxy,
            "xywh": xywh,
            "ori_image": origianl_image,
        }

    def collate_fn(self, data_list):
        batch = {"batch_idx": []}

        for i, data in enumerate(data_list):
            for key in data:
                values = batch.get(key, [])
                values.append(data[key])
                batch[key] = values
            batch["batch_idx"].append(np.array([i] * len(data["xyxy"])))
        batch["xywh"] = np.concatenate(batch["xywh"], axis=0)
        batch["xyxy"] = np.concatenate(batch["xyxy"], axis=0)
        batch["batch_idx"] = np.concatenate(batch["batch_idx"], axis=0)
        batch["image"] = np.stack(batch["image"], axis=0)

        return batch

    def shuffle(self):
        self.deck_data = self.make_deck_images()

    def preprocessing(self, image):
        image = image[:, :, ::-1]  # bgr -> rgb
        image = torch.from_numpy(image).float()
        image = torch.permute(image, [2, 0, 1])  # hwc -> chw
        return image
