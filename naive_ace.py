import numpy as np
import cv2 as cv
import pprint
import concurrent.futures

from numba import jit
from math import ceil
from PIL import Image

from utils import split_channels, get_window
from tqdm import tqdm


def im_val_exhaustive(x, y, img):
    sum = 0.0
    for xi in range(img.shape[0]):
        for yi in range(img.shape[1]):
            if x != xi or y != yi: 
                pix_dif = r(int(img[x][y]) - int(img[xi][yi]))
                dist = distance(x,y, xi,yi)
                sum += pix_dif / dist 
    return sum

def im_val_exhaustive_norm(x, y, img):
    sum = 0.0
    for xi in range(img.shape[0]):
        for yi in range(img.shape[1]):
            if x != xi or y != yi: 
                dist = distance(x,y, xi,yi)
                sum += 1/ dist 
    return sum


def im_val_window(x, y, img, window_size = 500):
    sum = 0.0
    wx, wy = get_window(x, y, img, window_size)
    for xi in range(wx[0], wx[1]):
        for yi in range(wy[0], wy[1]):
            if x != xi or y != yi: 
                pix_dif = r(int(img[x][y]) - int(img[xi][yi]))
                dist = distance(x,y, xi,yi)
                sum += pix_dif / dist    
    return sum


def fill_IM(IM, img):
    """
    Fill Intermediate Matrix
    """
    for x in tqdm(range(IM.shape[0])):
        for y in range(IM.shape[1]):
            # IM[x][y] = im_val_window(x, y, img, window_size= 4)
            # IM[x][y] = im_val_exhaustive(x, y, img)
            IM[x][y] = im_val_exhaustive(x, y, img) / im_val_exhaustive_norm(x, y, img)

def csa(img: Image) -> np.array:
    """
    Chromatic Spatial Adjustment 
    """
    IM = np.zeros(img.shape, dtype= np.float32)
    fill_IM(IM, img)
    return IM
    

def css(img):
    """
    Color Space Scaling
    """
    res = np.zeros(shape= img.shape, dtype=np.uint8)
    Max_IM = np.ndarray.max(img)
    Min_IM = np.ndarray.min(img)
    S  = 255/ (Max_IM - Min_IM)
    D_max = 255
    D_mid = D_max / 2
    for x in tqdm(range(img.shape[0])):
        for y in range(img.shape[1]):
            # res[x][y] = ceil(D_mid + (D_mid / Max_IM) * img[x][y]) # ?????
            # res[x][y] = ceil(D_mid + (D_mid / D_max) * img[x][y])
            res[x][y] = ceil(D_mid + (S) * img[x][y])
    return res

@jit
def r(pix_v, slope= 20) -> float:
    """
    contrast tuning function
    """
    # return 1 if pix_v > 0 else -1 
    if pix_v <= (-1/slope):
       return -1.0
    if(-1/slope) < pix_v < (1/slope):
        return float(pix_v * slope)
    if pix_v >= (1/slope):
        return 1.0

@jit
def distance(px1_x, px1_y, px2_x, px2_y):
    """
    return euclidean distance between two pixel
    """
    return np.sqrt((px1_x - px2_x) ** 2 + (px1_y - px2_y) ** 2)



def ace(img: Image):
    (b, g, r) = split_channels(img)
    cv.imshow("rO", r)
    cv.imshow("gO", g)
    cv.imshow("bO", b)

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

    
    cv.imshow("r", r_fin)
    cv.imshow("g", g_fin)
    cv.imshow("b", b_fin)
    cv.waitKey(0)
    return cv.merge((b_fin, g_fin, r_fin))


def main():
   
    img = cv.imread("/home/seb/Desktop/HdrProject/Dataset/stella/mid.jpg")
    img = cv.resize(img, (100, 100), interpolation= cv.INTER_AREA)
    cv.imshow("Resized", img)

    img = ace(img)

    cv.imshow("Hdr image",img)
    cv.imwrite("naive_ACE.jpg", img)
    cv.waitKey(0)    

main()

