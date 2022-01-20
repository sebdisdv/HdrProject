from cmath import inf
from typing import List
from black import re
import cv2 as cv
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from numba import jit
from tqdm import tqdm
from pprint import pprint
from utils import get_region_indexes


@jit
def P(v):
    return v / 255

@jit
def deltaIx(img, channel, x, y):

    res = 0
    if (x+1 < img.shape[0] and y < img.shape[1]):
        res = abs(int(img[x+1][y][channel]) - int(img[x][y][channel]))
    else:
        res = 0
    return res

@jit
def deltaIy(img, channel, x, y):
    res = 0
    if (y-1 > 0 and x < img.shape[0]):
        res = abs(int(img[x][y-1][channel]) - int(img[x][y][channel]))
    else:
        res = 0
    return res


def getDetailsRegions(imgs):
    region_indexes = get_region_indexes(imgs[0].shape[0], imgs[0].shape[1], 4)
    region_indexes = np.array(region_indexes) 
    M = []
    for i in range(len(imgs)): # immagini
        M.append([])
        for j in tqdm(range(region_indexes.shape[0])):# score di ogni regione di un immagine
            M_B = 0
            M_G = 0
            M_R = 0
            x_c, y_c = region_indexes[j][0]
            for x in range(region_indexes[j][0][0], region_indexes[j][0][1]):
                for y in range(region_indexes[j][1][0], region_indexes[j][1][1]):
                    M_B += P(max(deltaIx(imgs[i], 0, x, y), deltaIy(imgs[i], 0, x, y)))
                    M_G += P(max(deltaIx(imgs[i], 1, x , y), deltaIy(imgs[i], 1, x, y)))
                    M_R += P(max(deltaIx(imgs[i], 2, x, y), deltaIy(imgs[i], 2, x, y)))
            M[i].append([M_B, M_G, M_R]) # regione j in immagine i
        
    return np.array(M), region_indexes


def joinBestRegions(imgs, M, region_indexes):
    res = np.zeros(imgs[0].shape)
    for channel_indx in range(3):
        for r_indx in tqdm(range(M.shape[1])): # iterate over each region
            max_r = {}
            for i in range(len(imgs)):
                max_r[np.sum(M[i][r_indx])] = i
            index_image = max_r[max(max_r)]
            for i in range(region_indexes[r_indx][0][0], region_indexes[r_indx][0][1]):
                for j in range(region_indexes[r_indx][1][0], region_indexes[r_indx][1][1]):
                    res[i][j][channel_indx] = imgs[index_image][i][j][channel_indx]
    return res

@jit
def gaussianBlendingFunction(x, y, i, j):
    pass




def compute(imgs, exposures):
    M, regions_indexes = getDetailsRegions(imgs)
    # pprint(list(regions_indexes))
    # print(M.shape)
    res = joinBestRegions(imgs, M , regions_indexes)
    
    # cv.imwrite("asdasdasda.jpg", res)
    print(res.shape)
    
