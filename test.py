from numpy import uint8
import numpy as np
import cv2 as cv
import pprint
# ori = cv.imread("/home/seb/Desktop/HdrProject/Dataset/stella/mid.jpg")
# hdr = cv.imread("/home/seb/Desktop/HdrProject/naive_ACE.jpg")

# img = cv.resize(ori, (100, 100), interpolation= cv.INTER_AREA)
# cv.imwrite("/home/seb/Desktop/HdrProject/Dataset/stella/mid100x100.jpg", img)

l = [[0 for _ in range(10)] for _ in range(10)]

a = np.zeros(shape= (10,2), dtype=uint8)
b = np.ones(shape=(a.shape[0],1), dtype=uint8)
print(a)
print(b)
print(np.concatenate((a,b), axis=1))


