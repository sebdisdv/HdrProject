import cv2 as cv
from PIL.ExifTags import TAGS
from PIL.Image import Image


def get_exposure (img:Image) -> float:
   """
   This is a multiline comment
   """
   exif = img._getexif()
   for (k,v) in exif.items():
      if TAGS.get(k) == "ExposureTime":
         return v

def split_channels(img: Image):
   """
   Return (B,G,R) channels of the image
   """
   return cv.split(img)


def get_window(x, y, img, window_size):

   wx = [x - window_size if x- window_size > 0 else 0, x + window_size if x + window_size < img.shape[0] else img.shape[0] - 1]
   wy = [y - window_size if y - window_size > 0 else 0, y + window_size if y + window_size < img.shape[1] else img.shape[1] - 1]

   return wx, wy


