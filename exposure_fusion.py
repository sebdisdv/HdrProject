from math import exp
import cv2 as cv
from matplotlib.pyplot import axis
import numpy as np

from numba import jit
from pprint import pprint
from torch import le
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor



def get_contrast_weights(imgs, N):
    C = np.zeros((N, imgs[0].shape[0], imgs.shape[1]), dtype= np.float32)
    kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype= np.float32)
    for i in range(N):
        gray = cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY)
        C[i] = abs(cv.filter2D(gray, -1, kernel)) 
    return C

def get_saturation_weights(imgs, N):
    S = np.zeros((N, imgs[0].shape[0], imgs.shape[1]), dtype= np.float32)
    for i in range(N):
        b, g, r = cv.split(imgs[i])
        S[i] = np.std([b, g, r], axis=0)

    return S

def get_well_exposedness_weights(imgs, N):
    sigma = 0.2 ** 2
    E = np.zeros((N, imgs[0].shape[0], imgs.shape[1]), dtype= np.float32)
    for i in range(N):
        b, g, r = cv.split(imgs[i])
        B = np.exp(-0.5*np.power(b - 0.5, 2)/sigma)
        G = np.exp(-0.5*np.power(g - 0.5, 2)/sigma)
        R = np.exp(-0.5*np.power(r - 0.5, 2)/sigma)
        E[i] = np.multiply(B, np.multiply(G, R))
    return E


def get_weights_map(imgs):
    N = len(imgs) # number of images
    W = np.ones((N, imgs[0].shape[0], imgs[0].shape[1]), dtype= np.float32)
    W = np.multiply(W, get_contrast_weights(imgs, N))
    W = np.multiply(W, get_saturation_weights(imgs, N))
    W = np.multiply(W, get_well_exposedness_weights(imgs, N))

    #normalize

    sum = W.sum(axis= 0)
    W = np.divide(W, sum+ 1e-12)
    # np.seterr(divide='ignore', invalid='ignore')

    return W

def compute(imgs, exposures):
    
    for i in range(len(imgs)):
        imgs[i] = np.float32(imgs[i]) / 255.0
    
    W = get_weights_map(imgs)