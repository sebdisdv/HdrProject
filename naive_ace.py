import numpy as np
import cv2 as cv
import pprint

from numba import jit
from math import ceil
from PIL import Image

from utils import split_channels
from tqdm import tqdm



# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()

# possible OPTIMIZATION
# Reduce the area of im_val


@jit
def im_val(x, y, img):
    sum = 0.0
    for xi in range(img.shape[0]):
        for yi in range(img.shape[1]):
            if x != xi and y != yi: 
                pix_dif = r(img[x][y] - img[xi][yi])
                dist = distance(x,y, xi,yi)
                sum += pix_dif / dist 
            
    return sum


def fill_IM(IM, img):
    for x in range(IM.shape[0]):
        for y in tqdm(range(IM.shape[1])):
            IM[x][y] = im_val(x, y, img)
    
def csa(img: Image) -> np.array:
    """
    Chromatic Spatial Adjustment 
    """
    IM = np.zeros(img.shape, dtype= np.float32)
    fill_IM(IM, img)
    # for x in range(IM.shape[0]):
        # for y in tqdm(range(IM.shape[1])):
        #     IM[x][y] = im_val(x, y, img)
        # pr.disable()
        # with open("log.txt", "w") as f:
        #     s = io.StringIO()
        #     sortby = SortKey.CUMULATIVE
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     f.write(s.getvalue())
        # breakpoint()
    return IM
    

def css(img):
    """
    Color Space Scaling
    """
    res = np.zeros(shape= img.shape, dtype=np.int8)
    Max_IM = max(img)
    D_max = 255
    D_mid = D_max / 2
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            res[x][y] = ceil(D_mid + (D_mid / Max_IM) * img[x][y])
    return res

@jit
def r(pix_v, slope= 1) -> float:
    """
    contrast tuning function
    """
    if pix_v <= (-1/slope):
        return -1
    elif (-1/slope) < pix_v < (1/slope):
        return pix_v * slope
    elif pix_v >= (1/slope):
        return 1

@jit
def distance(px1_x, px1_y, px2_x, px2_y):
    """
    return euclidean distance between two pixel
    """
    return np.sqrt((px1_x - px2_x) ** 2 + (px1_y - px2_y) ** 2)


def ace(img: Image):
    b, g, r = split_channels(img)
    b = np.int16(b)
    g = np.int16(g)
    r = np.int16(r)

    b_im = csa(b)
    g_im = csa(g)
    r_im = csa(r)

    b_fin = css(b)
    g_fin = css(g)
    r_fin = css(r)

    return cv.join([b, g, r])
    # pprint.pprint(im1)

def main():
    # pr.enable()
    img = cv.imread("/home/seb/Desktop/HdrProject/Dataset/stella/mid.jpg")
    img = cv.resize(img, (1920, 1080), interpolation= cv.INTER_AREA)
    img = ace(img)
    cv.imshow("Hdr image",img)
    cv.waitkey(0)    

main()