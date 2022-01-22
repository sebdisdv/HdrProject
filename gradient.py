
from math import exp
from typing import List
from black import re
import cv2 as cv
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from numba import jit
from numpy import float32
from tqdm import tqdm
from pprint import pprint, pformat
from utils import get_region_indexes, get_region_centers, associate_index_to_centers, get_window


@jit
def P(v):
    return v / 255


@jit
def deltaIx(img, channel, x, y):

    res = 0
    if x + 1 < img.shape[0] and y < img.shape[1]:
        res = abs(int(img[x + 1][y][channel]) - int(img[x][y][channel]))
    else:
        res = 0
    return res


@jit
def deltaIy(img, channel, x, y):
    res = 0
    if y - 1 > 0 and x < img.shape[0]:
        res = abs(int(img[x][y - 1][channel]) - int(img[x][y][channel]))
    else:
        res = 0
    return res


def getDetailsRegions(imgs):
    region_indexes = get_region_indexes(imgs[0].shape[0], imgs[0].shape[1], 5)
    # region_indexes = np.array(region_indexes)
    M = []
    for i in range(len(imgs)):  # immagini
        M.append([])
        for j in tqdm(
            range(region_indexes.shape[0])
        ):  # score di ogni regione di un immagine
            M_B = 0
            M_G = 0
            M_R = 0
            # x_c, y_c = region_indexes[j][0]
            for x in range(region_indexes[j][0][0], region_indexes[j][0][1]):
                for y in range(region_indexes[j][1][0], region_indexes[j][1][1]):
                    M_B += P(max(deltaIx(imgs[i], 0, x, y), deltaIy(imgs[i], 0, x, y)))
                    M_G += P(max(deltaIx(imgs[i], 1, x, y), deltaIy(imgs[i], 1, x, y)))
                    M_R += P(max(deltaIx(imgs[i], 2, x, y), deltaIy(imgs[i], 2, x, y)))
            M[i].append([M_B, M_G, M_R])  # regione j in immagine i

    return np.array(M), region_indexes


def joinBestRegions(imgs, M, region_indexes):
    res = np.zeros(imgs[0].shape)
    for channel_indx in range(3):
        for r_indx in tqdm(range(M.shape[1])):  # iterate over each region
            max_r = {}
            for i in range(len(imgs)):
                max_r[np.sum(M[i][r_indx])] = i
            index_image = max_r[max(max_r)]
            for i in range(region_indexes[r_indx][0][0], region_indexes[r_indx][0][1]):
                for j in range(
                    region_indexes[r_indx][1][0], region_indexes[r_indx][1][1]
                ):
                    res[i][j][channel_indx] = imgs[index_image][i][j][channel_indx]
    
    return res


@jit
def U(x_c_reg, y_c_reg, x_c, y_c):
    epsilon = 30
    return abs(int(x_c_reg) - int(x_c)) <= epsilon and abs(int(y_c_reg) - int(y_c)) <= epsilon


@jit
def exp_g(x, y, x_c, y_c) -> float:
    sigma_x = 60
    sigma_y = 60
    return exp(
        -((((int(x) - int(x_c)) ** 2) / (2 * sigma_x)) + (((int(y) - int(y_c)) ** 2) / (2 * sigma_y)))
    )

@jit
def gaussianBlendingFunction(x, y, x_c, y_c, region_indexes, center_indexes):

    num = exp_g(x, y, x_c, y_c)
    den = 0.0
    for i in range(center_indexes.shape[0]):
        den += exp_g(x, y, center_indexes[i][0], center_indexes[i][1])
    
    den *= center_indexes.shape[0]
    
    return num / den



def compute_channel(channel, region_indexes, center_indexes, map_px_center):
    res = np.zeros(shape=channel.shape, dtype=float32)
    for x in tqdm(range(res.shape[0])):
        for y in range(res.shape[1]):
            window = get_window(x, y, channel, 10) # WINDOW VERSION
            for i in range(window[0][0], window[0][1]):
                for j in range(window[1][0], window[1][1]):
            # for i in range(res.shape[0]):
            #     for j in range(res.shape[1]):
                    add = 0
                    if U(
                        map_px_center[(i, j)][0],
                        map_px_center[(i, j)][1],
                        map_px_center[(x, y)][0], 
                        map_px_center[(x, y)][1]
                        ):
                        add = 1
                        add *= gaussianBlendingFunction(
                            map_px_center[(x, y)][0], 
                            map_px_center[(x, y)][1],
                            map_px_center[(i, j)][0],
                            map_px_center[(i, j)][1],
                            region_indexes,
                            center_indexes
                        )
                        add *= channel[x][y]
                    res[x][y] += add
    return res


def blend(img, regions_indexes):
    centers_indexes = get_region_centers(regions_indexes)
    pixel_region_center = associate_index_to_centers(regions_indexes, centers_indexes)
    b, g, r = cv.split(img)

    with ProcessPoolExecutor() as excecutor:
        proc1 = excecutor.submit(
            compute_channel, b, regions_indexes, centers_indexes, pixel_region_center
        )
        proc2 = excecutor.submit(
            compute_channel, g, regions_indexes, centers_indexes, pixel_region_center
        )
        proc3 = excecutor.submit(
            compute_channel, r, regions_indexes, centers_indexes, pixel_region_center
        )

    b = proc1.result()
    g = proc2.result()
    r = proc3.result()

    return cv.merge((b, g, r))


def compute(imgs):
    
    for i in range(len(imgs)): # Values of intensity must be between 0 and 1
        imgs[i] = imgs[i] / 255.0

    M, regions_indexes = getDetailsRegions(imgs)
    # pprint(list(regions_indexes))
    # print(regions_indexes.shape)
    # res = joinBestRegions(imgs, M, regions_indexes)
    # res = res / 255.0
    # pprint(res)
    # cv.imwrite("res_Middle_Gradient.jpg", (res / np.amax(res)) * 255)
    # center_indexes = get_region_centers(regions_indexes)
    # pixel_region_center = associate_index_to_centers(regions_indexes, center_indexes)
    # pprint(pixel_region_center)
    # exit()
    res = blend(joinBestRegions(imgs, M, regions_indexes), regions_indexes)
    
    res = res / np.amax(res)
    res = 255 * res
    # pprint(res)
    # pprint(res)
    # pprint(list(get_region_centers(regions_indexes)))
    cv.imwrite("res__Final_Gradient.jpg", res)

    

    # print(res.shape)
    return res
