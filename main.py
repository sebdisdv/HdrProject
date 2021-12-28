import os.path as path
import os
import cv2
from PIL import Image, ExifTags
import numpy as np
from utils import get_exposure

images = ["HdrProject/Dataset/stella/under.jpg", "HdrProject/Dataset/stella/mid.jpg", "HdrProject/Dataset/stella/over.jpg"]
exposure_times = np.array([get_exposure(Image.open(img)) for img in images], dtype= np.float32)

images = [cv2.imread(x) for x in images]

merge_debevec = cv2.createMergeDebevec()
hdr_debevec = merge_debevec.process(images, times= exposure_times.copy())



tonemap = cv2.createTonemapDrago()
res_debevec = tonemap.process(hdr_debevec.copy())

res = np.clip(res_debevec* 255, 0, 255).astype('uint8')

cv2.imwrite("res.jpg", res)

