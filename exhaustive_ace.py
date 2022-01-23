import concurrent.futures
from math import ceil

import cv2 as cv
import numpy as np
from numba import jit
from PIL import Image
from tqdm import tqdm

from utils import split_channels


@jit
def r(pix_v, slope=20) -> float:
    """
    contrast tuning function
    """
    # return 1 if pix_v > 0 else -1
    if pix_v <= (-1 / slope):
        return -1.0
    if (-1 / slope) < pix_v < (1 / slope):
        return pix_v * slope
    if pix_v >= (1 / slope):
        return 1.0


@jit
def distance(px1_x, px1_y, px2_x, px2_y):
    """
    return euclidean distance between two pixel
    """
    return np.sqrt((px1_x - px2_x) ** 2 + (px1_y - px2_y) ** 2)


@jit
def imVal(x, y, img):
    sum = 0.0
    norm_factor = 0.0
    for xi in range(img.shape[0]):
        for yi in range(img.shape[1]):
            # if x != xi or y != yi:
            pix_dif = r(img[x][y] - img[xi][yi])
            dist = distance(x, y, xi, yi)
            if dist != 0:
                norm_factor += 1 / dist
                sum += pix_dif / dist
    return sum / norm_factor


def fill_IM(IM, img):
    """
    Fill Intermediate Matrix
    """
    for x in tqdm(range(IM.shape[0])):
        for y in range(IM.shape[1]):
            IM[x][y] = imVal(x, y, img)


def csa(img: Image) -> np.array:
    """
    Chromatic Spatial Adjustment
    """
    IM = np.zeros(img.shape, dtype=np.float32)
    fill_IM(IM, img)
    return IM


def css(img):
    """
    Color Space Scaling
    """
    res = np.zeros(shape=img.shape, dtype=np.float32)
    Max_IM = np.amax(img)
    Min_IM = np.amin(img)
    S = 255 / (Max_IM - Min_IM)
    D_max = 255
    D_mid = D_max / 2
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            res[x][y] = ceil(D_mid + (S) * img[x][y])
    return res


def ace(img: Image):
    (b, g, r) = split_channels(img)

    b = np.float32(b)
    g = np.float32(g)
    r = np.float32(r)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        proc1 = executor.submit(csa, b)
        proc2 = executor.submit(csa, g)
        proc3 = executor.submit(csa, r)

        b_im = proc1.result()
        g_im = proc2.result()
        r_im = proc3.result()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        proc1 = executor.submit(css, b_im)
        proc2 = executor.submit(css, g_im)
        proc3 = executor.submit(css, r_im)

        b_fin = proc1.result()
        g_fin = proc2.result()
        r_fin = proc3.result()

    return cv.merge((b_fin, g_fin, r_fin))


def compute(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
    img = ace(img)

    return img
