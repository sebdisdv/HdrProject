import cv2 as cv
import numpy as np
from PIL.ExifTags import TAGS
from PIL.Image import Image
from random import sample


def get_exposure (img:Image) -> float:
   """
   This is a multiline comment
   """
   exif = img._getexif()
   for (k,v) in exif.items():
      if TAGS.get(k) == "ExposureTime":
         return v[0]/v[1]

def split_channels(img: Image):
   """
   Return (B,G,R) channels of the image
   """
   return cv.split(img)


def get_window(x, y, img, window_size):

   wx = [x - window_size if x - window_size > 0 else 0, x + window_size if x + window_size < img.shape[0] else img.shape[0] - 1]
   wy = [y - window_size if y - window_size > 0 else 0, y + window_size if y + window_size < img.shape[1] else img.shape[1] - 1]

   return wx, wy


def get_pixels_indexes(img_shape):
   Z_indexes = np.zeros(shape= (500, 2), dtype= np.uint8)
   x_idxs = sample(range(img_shape[0]), k= 50)
   y_idxs = range(img_shape[1])
   k= 0
   for i in x_idxs:
      for j in sample(y_idxs, k= 10):
         Z_indexes[k][0] = i
         Z_indexes[k][1] = j
         k += 1
   return Z_indexes

