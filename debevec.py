from pprint import pprint
import cv2 as cv
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

from numba import jit
from tqdm import tqdm


from math import log
from utils import get_pixels_indexes

@jit
def w(z):
    if z <= (0 + 255) / 2:
        return z - 0 + 1
    else:
        return 255 - z + 1

    

#TODO Scegliere meglio i pixel!!
def compute_Z(channel):
    Z_indexes = get_pixels_indexes((channel.shape[1], channel.shape[2]))
    # 100 is the number of pixel necessary
    Z = np.zeros(shape= (500, 3), dtype= np.uint8)
    k = 0
    for (i,j) in Z_indexes:
        Z[k][0] = channel[0][i][j]
        Z[k][1] = channel[1][i][j]
        Z[k][2] = channel[2][i][j]
        k+=1
    return Z

    

def debevec(channel, exposures):
    n = 256
    l = 50 # weight the smoothnees term relative to the data fitting term
    Z = compute_Z(channel)
    A = np.zeros(shape= (Z.shape[0] * Z.shape[1] + n + 1, Z.shape[0] + n), dtype=np.float32)
    b = np.zeros(shape= (A.shape[0], 1), dtype=np.float32)
    
    # Datafitting equations

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w(Z[i][j])
            A[k][Z[i][j]] = wij
            A[k][n+i] = -wij
            b[k] = wij * log(exposures[j])
            k += 1
    
    # Fix curve by setting its middle value to 0

    A[k, 129] = 0
    k += 1

    # include smoothness operation

    for i in range(1, n - 2):
        A[k][i] = l * w(i+1)
        A[k][i + 1] = -2 * l * w(i+1)
        A[k][i + 2] = l * w(i+1)
        k += 1
    # x = np.linalg.svd(np.concatenate((A , b), axis=1))[1]
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    g = x[0 : n]
    lnE = x[n: len(x)]

    # lnE = []
    # for i in range(Z.shape[0]):
    #     num = 0.0
    #     den = 0.0
    #     for j in range(Z.shape[1]):
    #         num += w(Z[i][j]) * (g[Z[i][j]][0]- log(exposures[j]))
    #         den += w(Z[i][j])
    #     lnE.append(num / den)
    # print(len(lnE))
    # print(len(g))
    # print(len(lnE))
    
    
    return g, lnE




class DebevecResults():

    def __init__(self, res):
        self.g = res[0]
        self.lnE = res[1]

@jit
def calc_pxHdr(i, j, channel, g, exposures):
    num = 0
    den = 0
    for k in range(len(channel)):
        num += w(channel[k][i][j]) * (g[channel[k][i][j]][0] - log(exposures[k]))
        den += w(channel[k][i][j])
    return num / den

def recoverHdrRadianceMap(debevec_res: DebevecResults, channel, exposures):
    exposures = np.array(exposures)
    hdrMap = np.zeros(shape= (channel.shape[1], channel.shape[2]), dtype= np.float32)
    for i in tqdm(range(hdrMap.shape[0])):
        for j in range(hdrMap.shape[1]):
            hdrMap[i][j] = calc_pxHdr(i, j, channel, debevec_res.g, exposures)
    
    hdrMap = np.exp(hdrMap)
    hdrMap = hdrMap / np.amax(hdrMap)
    hdrMap = hdrMap * 255
    return hdrMap

def compute(imgs, exposures):
    b = np.zeros(shape= (imgs[0].shape[2], imgs[0].shape[0], imgs[0].shape[1]), dtype= np.uint8)
    g = np.zeros(shape= (imgs[0].shape[2], imgs[0].shape[0], imgs[0].shape[1]), dtype= np.uint8)
    r = np.zeros(shape= (imgs[0].shape[2], imgs[0].shape[0], imgs[0].shape[1]), dtype= np.uint8)
    i = 0
    for img in imgs:
        (bi, gi, ri) = cv.split(img)
        b[i] = bi
        g[i] = gi
        r[i] = ri
        i += 1

    # print(b)
   
    with concurrent.futures.ProcessPoolExecutor() as excecutor:
        proc1 = excecutor.submit(debevec, b, exposures)
        proc2 = excecutor.submit(debevec, g, exposures)
        proc3 = excecutor.submit(debevec, r, exposures)

    b_res = DebevecResults(proc1.result())
    g_res = DebevecResults(proc2.result())
    r_res = DebevecResults(proc3.result())

    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        proc1 = executor.submit(recoverHdrRadianceMap, b_res, b, exposures)
        proc2 = executor.submit(recoverHdrRadianceMap, g_res, g, exposures)
        proc3 = executor.submit(recoverHdrRadianceMap, r_res, r, exposures)

    b_hdr = proc1.result()
   
    # b_hdr = np.exp(b_hdr)
    # print(np.amax(b_hdr))
    g_hdr = proc2.result()
    r_hdr = proc3.result()

    hdr = cv.merge((b_hdr, g_hdr, r_hdr))
    
    
    
    # hdr = ((hdr/np.amax(hdr))).astype(np.float32)
    # plt.imshow(cv.cvtColor(hdr, cv.COLOR_BGR2RGB))
    # plt.imsave("HDRFORSE.png", cv.cvtColor(hdr, cv.COLOR_BGR2RGB))
    # plt.plot(range(255), b_res.g, 'ro')
    # plt.savefig("PLOT_norm.png")
    plt.plot(b_res.g, range(256))
    
    # plt.axis([-5,5, 0 , 300])
    plt.savefig("PLOT_B.png")
    
    plt.plot(g_res.g, range(256))
    # plt.axis([-5,5, 0 , 300])
    plt.savefig("PLOT_G.png")
    
    plt.plot(r_res.g, range(256))
    # plt.axis([-5,5, 0 , 300])
    plt.savefig("PLOT_R.png")
    
    # pprint(d)
    
    # return cv.merge((b, g, r))
    return hdr


