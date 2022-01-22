import cv2 as cv
import numpy as np
from tqdm import tqdm


def get_contrast_weights(imgs, N):
    C_weights = np.zeros((N, imgs[0].shape[0], imgs[0].shape[1]), dtype=np.float32)
    kernel = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
    )  # laplacian filter
    for i in range(N):
        gray = cv.cvtColor(
            imgs[i], cv.COLOR_BGR2GRAY
        )  # contrast must be computed on the gray image
        C_weights[i] = abs(
            cv.filter2D(gray, -1, kernel)
        )  # we apply the laplacian filter on the gray scale image
    return C_weights


def get_saturation_weights(imgs, N):
    S_weights = np.zeros((N, imgs[0].shape[0], imgs[0].shape[1]), dtype=np.float32)
    for i in range(N):
        b, g, r = cv.split(imgs[i])
        S_weights[i] = np.std(
            [b, g, r], axis=0
        )  # standard deviation within the channels at each pixel

    return S_weights


def get_well_exposedness_weights(imgs, N):
    sigma = 0.5 ** 2  # value used in the paper
    E_weights = np.zeros((N, imgs[0].shape[0], imgs[0].shape[1]), dtype=np.float32)
    for i in range(N):
        b, g, r = cv.split(
            imgs[i]
        )  # each channel must be computed separately and them multiply all together
        B = np.exp(-0.5 * np.power(b - 0.5, 2) / sigma)  # blue channel
        G = np.exp(-0.5 * np.power(g - 0.5, 2) / sigma)  # green channel
        R = np.exp(-0.5 * np.power(r - 0.5, 2) / sigma)  # red channel
        E_weights[i] = np.multiply(B, np.multiply(G, R))
    return E_weights


def get_weights_map(imgs):
    N = len(imgs)  # number of images
    W = np.ones((N, imgs[0].shape[0], imgs[0].shape[1]), dtype=np.float32)

    # In our case the exponent of each type of weight is 1
    W = np.multiply(W, get_contrast_weights(imgs, N))  # element wise multiplication
    W = np.multiply(W, get_saturation_weights(imgs, N))
    W = np.multiply(W, get_well_exposedness_weights(imgs, N))

    # normalize

    sum = W.sum(axis=0)
    W = np.divide(W, sum + 1e-12)
    np.seterr(divide="ignore", invalid="ignore")

    return W


def get_Gaussian_pyramid_weight_map(w, pyr_levels):
    w_copy = w.copy()
    pyrWeights = [w_copy]
    for i in range(1, pyr_levels):
        w_copy = cv.pyrDown(cv.GaussianBlur(w_copy, (3, 3), 0))
        pyrWeights.append(w_copy)
    return pyrWeights


def get_Laplacian_pyramid_image(img, pyr_levels):
    pyrLaplacian = []

    img_cp = img.copy()
    for i in range(0, pyr_levels - 1):
        src = cv.GaussianBlur(img_cp, (3, 3), 0)
        img = cv.pyrDown(src)
        up_lv = cv.pyrUp(img)
        temp = cv.resize(up_lv, (img_cp.shape[:2][1], img_cp.shape[:2][0]))
        new_level = cv.subtract(img_cp, temp)
        pyrLaplacian.append(new_level)
        img_cp = img.copy()

    pyrLaplacian.append(img)

    return pyrLaplacian


# Construct Laplacian pyramid
def get_pyramid(imgs, N, W, level_pyr=8):

    # init pyramid levels for reconstruction
    w = imgs[0].shape[0]
    h = imgs[0].shape[1]
    pyramid = [np.zeros(shape=(w, h, 3), dtype=np.float32)]
    for _ in range(level_pyr):
        w = int(np.ceil(w / 2))
        h = int(np.ceil(h / 2))
        pyramid.append(np.zeros(shape=(w, h, 3), dtype=np.float32))

    # construct pyramid
    for i in range(N):
        pyrW = get_Gaussian_pyramid_weight_map(W[i], level_pyr)
        pyrI = get_Laplacian_pyramid_image(imgs[i], level_pyr)

        for l in range(level_pyr):
            b = np.zeros(shape=(len(pyrW[l]), len(pyrI[l][0]), 3), dtype=np.float32)
            b[:, :, 0] = pyrW[l]
            b[:, :, 1] = pyrW[l]
            b[:, :, 2] = pyrW[l]
            pyramid[l] += np.multiply(b, pyrI[l])

    return pyramid


def compute(imgs):

    level_pyr = 12

    for i in range(len(imgs)):  # Values of intensity must be in the range [0...255]
        imgs[i] = np.float32(imgs[i]) / 255.0

    W = get_weights_map(imgs)
    pyr = get_pyramid(imgs, len(imgs), W, level_pyr)

    res = pyr[level_pyr - 1].copy()
    for l in tqdm(
        range(level_pyr - 2, -1, -1)
    ):  # pyramid is collapsed up to the original image size
        res = pyr[l] + cv.resize(
            cv.pyrUp(res), (pyr[l].shape[:2][1], pyr[l].shape[:2][0])
        )

    # value of the resulting imagare are mapped in the range [0...255]
    res = res / np.amax(res)
    res = res * 255

    return res