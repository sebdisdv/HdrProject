import cv2 as cv
from matplotlib.pyplot import axes
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

from math import log
from utils import get_pixels_indexes

def w(z, Z_min= 0, Z_max = 255):
    if z <= 1/2 * (Z_min + Z_max):
        return z - Z_min
    else:
        return Z_max - z

def compute_Z(channel):
    Z_indexes = get_pixels_indexes((channel.shape[1], channel.shape[2]))
    # 100 is the number of pixel necessary
    Z = np.zeros(shape= (100, 3), dtype= np.uint8)
    k = 0
    for (i,j) in Z_indexes:
        Z[k][0] = channel[0][i][j]
        Z[k][1] = channel[1][i][j]
        Z[k][2] = channel[2][i][j]
    return Z

    

def debevec(channel, exposures):
    Z_min = 0
    Z_max = 255
    n = 255
    l = 10 # weight the smoothnees term relative to the data fitting term
    Z = compute_Z(channel)
    A = np.zeros(shape= (Z.shape[0] * Z.shape[1] + n + 1, Z.shape[0] + n))
    b = np.zeros(shape= (A.shape[0], 1))

    # Datafitting equations

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w(Z[i][j] + 1)
            A[k][Z[i][j] + 1] = wij
            A[k][n+i] = -wij
            b[k][0] = wij * exposures[j]
            k += 1
    
    # Fix curve by setting its middle value to 0

    A[k, 129] = 1
    k += 1

    # include smoothness operation

    for i in range(0, n - 3):
        A[k][i] = l * w(i+1)
        A[k][i + 1] = -2 * l * w(i+1)
        A[k][i + 2] = l * w(i+1)
        k += 1
    # x = np.linalg.svd(np.concatenate((A , b), axis=1))[1]
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[0: n]
    lnE = x[n: len(x) + 1]
    
    print(len(g))
    print(len(lnE))
    

    return g, lnE

def compute(imgs, exposures):
    b = np.zeros(shape= (imgs[0].shape[2], imgs[0].shape[0], imgs[0].shape[1]), dtype= np.uint8)
    # g = []
    # r = []
    i = 0
    for img in imgs:
        (bi, gi, ri) = cv.split(img)
        b[i] = bi
        i += 1
        # g.append(gi)
        # r.append(ri)
    
    # print(b)
   
    # with concurrent.futures.ProcessPoolExecutor() as excecutor:
    #     proc1 = excecutor.submit(debevec, b, exposures)
        # proc2 = excecutor.submit(debevec, g, exposures)
        # proc3 = excecutor.submit(debevec, r, exposures)

    #     b = proc1.result()
        # g = proc2.result()
        # r = proc3.result()

    c, d = debevec(b, exposures)
    
    plt.plot(c, range(255))
    plt.savefig("PLOT.png")
    # return cv.merge((b, g, r))



