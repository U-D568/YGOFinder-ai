import random

import cv2
import math
import numpy as np

from utils import common


def random_pixelate(image, min_scale=0.5, max_scale=1.0):
    height, width = image.shape[:2]

    scale = random.uniform(min_scale, max_scale)
    new_height = round(height * scale)
    new_width = round(width * scale)

    # zoom-in and zoom-out
    pixelated = cv2.resize(image, (new_width, new_height))
    pixelated = cv2.resize(pixelated, (width, height))

    return pixelated


def random_zoom_transition(image, min_scale=0.5, max_scale=1.0):
    # resize image
    height, width = image.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    new_height = round(height * scale)
    new_width = round(width * scale)
    new_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # resize and put padding
    dw = width - new_width
    dh = height - new_height
    x_pos = round(dw * random.random())
    y_pos = round(dh * random.random())
    new_image = noise_background(height, width, new_image, (x_pos, y_pos))

    return new_image, scale, (x_pos, y_pos)


def noise_background(bg_height, bg_width, image, xy_pos):
    im_h, im_w = image.shape[:2]
    noise = np.random.normal(loc=127, scale=30, size=(bg_height, bg_width, 3))
    noise[xy_pos[1] : xy_pos[1] + im_h, xy_pos[0] : xy_pos[0] + im_w] = image
    return noise


def make_square_shape(image: np.array, target_size: int):  # size: image size to convert
    height, width = image.shape[:2]
    ratio = target_size / max(height, width)  # ratio
    if ratio != 1:  # if sizes are not equal
        width = min(math.ceil(width * ratio), target_size)
        height = min(math.ceil(height * ratio), target_size)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # padding noise
    dw = (target_size - width) / 2
    dh = (target_size - height) / 2
    y_pos = int(round(dh - 0.1))
    x_pos = int(round(dw - 0.1))
    image = noise_background(target_size, target_size, image, (x_pos, y_pos))

    return image, ratio, (x_pos, y_pos)


def make_deck_image(images, padding_value=(114, 114, 114)):
    length = len(images)
    divisor = common.get_factors(length)

    random.shuffle(divisor)
    while True:
        row = divisor.pop()
        col = length // row
        ratio = max(row, col) / min(row, col)
        if ratio < 10:
            break
        if len(divisor) == 0:
            raise ValueError(f"{length}")

    height, width = images[0].shape[:2]
    sx = np.arange(0, col * width, width)
    sy = np.arange(0, row * height, height)
    sx, sy = np.meshgrid(sx, sy)
    offset = np.stack([sx, sy], axis=-1).reshape(-1, 2)
    if len(images) < length:
        diff = length - len(images)
        offset = offset[:-diff]

    total_image = []
    for row_index in range(row):
        start = row_index * col
        end = (row_index + 1) * col
        row_images = images[start:end]
        if len(row_images) < col:
            padding_image = np.full(images[0].shape, padding_value, dtype=np.uint8)
            num_padding = col - len(row_images)
            for _ in range(num_padding):
                row_images.append(padding_image.copy())
        total_image.append(np.concatenate(row_images, axis=1))

    return np.concatenate(total_image, axis=0), offset
