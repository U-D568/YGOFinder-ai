from collections import deque
import math
import os
import threading
import urllib

import cv2
import numpy as np

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def extract_images(path):
    images = []
    for curdir, _, files in os.walk(path):
        for file in files:
            _, ext = os.path.splitext(file)
            if not ext in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(curdir, file)
            images.append(path)
    return images


def get_filename(path):
    basename = os.path.basename(path)
    name, ext = os.path.splitext(basename)
    return name


def xyxy2xywh(ary: np.array):
    shape = ary.shape
    assert shape[-1] == 4, "Invalid input shape. The last dimension size have to be 4."

    flatten = ary.reshape(-1, 4)
    height = flatten[:, 2] - flatten[:, 0]
    width = flatten[:, 3] - flatten[:, 1]

    flatten[:, 0] += width / 2
    flatten[:, 1] += height / 2
    flatten[:, 2] = width
    flatten[:, 3] = height

    xywh = np.reshape(flatten, shape)
    return xywh


def xywh2xyxy(ary: np.array):
    shape = ary.shape
    assert shape[-1] == 4, "Invalid input shape. The last dimension size have to be 4."

    flatten = ary.reshape(-1, 4)
    center = flatten[:, :2]
    wh = flatten[:, 2:]
    flatten[:, :2] = center - wh
    flatten[:, 2:] = center + wh

    xyxy = np.reshape(flatten, shape)
    return xyxy


def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def get_factors(num):
    result = []
    limit = int(math.sqrt(num)) + 1
    for i in range(1, limit):
        if num % i == 0:
            result.append(i)
            result.append(num // i)
    return result


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
