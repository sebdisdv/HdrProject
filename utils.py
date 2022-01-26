import os
import cv2 as cv
import numpy as np
from PIL.ExifTags import TAGS
from PIL.Image import Image
from random import sample
from tqdm import tqdm
from os import path

def get_exposure(img: Image) -> float:
    """
    This is a multiline comment
    """
    exif = img._getexif()
    for (k, v) in exif.items():
        if TAGS.get(k) == "ExposureTime":
            if isinstance(v, tuple):
                return v[0] / v[1] 
            else:
                return v


def split_channels(img: Image):
    """
    Return (B,G,R) channels of the image
    """
    return cv.split(img)


def get_window(x, y, img, window_size):

    wx = [
        x - window_size if x - window_size > 0 else 0,
        x + window_size if x + window_size < img.shape[0] else img.shape[0] - 1,
    ]
    wy = [
        y - window_size if y - window_size > 0 else 0,
        y + window_size if y + window_size < img.shape[1] else img.shape[1] - 1,
    ]

    return np.array([wx, wy])


def get_pixels_indexes(img_shape):
    Z_indexes = np.zeros(shape=(500, 2), dtype=np.uint8)
    x_idxs = sample(range(img_shape[0]), k=50)
    y_idxs = range(img_shape[1])
    k = 0
    for i in x_idxs:
        for j in sample(y_idxs, k=10):
            Z_indexes[k][0] = i
            Z_indexes[k][1] = j
            k += 1
    return Z_indexes


# onlywork if width and height can be divided by window
def get_region_indexes(witdh, height, window):
    indexes = []
    for i in range(0, witdh, window):
        for j in range(0, height, window):
            indexes.append(((i, i + window), (j, j + window)))
    return np.array(indexes)


def get_region_centers(region_indexes):
    centers = np.zeros(shape=(region_indexes.shape[0], 2), dtype=np.uint)
    for i in range(centers.shape[0]):
        centers[i][0] = (
            region_indexes[i][0][1] - region_indexes[i][0][0]
        ) // 2 + region_indexes[i][0][0]
        centers[i][1] = (
            region_indexes[i][1][1] - region_indexes[i][1][0]
        ) // 2 + region_indexes[i][1][0]
    return np.array(centers)


def associate_index_to_centers(region_indexes, centers):
    res = {}
    for i in tqdm(range(region_indexes.shape[0])):
        for x in range(region_indexes[i][0][0], region_indexes[i][0][1]):
            for y in range(region_indexes[i][1][0], region_indexes[i][1][1]):
                res[(x, y)] = np.array(centers[i])
    return res

def get_dataset_info() -> tuple:
    names = os.listdir("./Dataset")
    info = {}
    for name in names:
        for _, _, imgs_path in os.walk(path.join("Dataset", name)):
            info[name] = imgs_path
    return names, info

def create_folders(names) -> None:
    if path.exists("Results"):
        for name in names:
            if not path.exists(path.join("Resutls", name)):
                os.mkdir(path.join("Results", name))
    else:
        os.mkdir("Results")
        for name in names:
            os.mkdir(path.join("Results", name))   