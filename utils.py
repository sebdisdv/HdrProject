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