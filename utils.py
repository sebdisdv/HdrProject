from PIL.ExifTags import TAGS

def get_exposure (img):
   exif = img._getexif()
   for (k,v) in exif.items():
      if TAGS.get(k) == "ExposureTime":
         return v