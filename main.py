import cv2
import json
import os.path as path
import numpy as np
import windowed_ace
import exhaustive_ace
from utils import get_exposure
from PIL import Image





# create folder if it does not exists 
# save inside it
# Create a setup file?


class HdrImplementations():

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.settings = json.load(open("settings.json"))
        self.images_paths = self.settings["dataset"][dataset_name]
        self.images = [cv2.imread(im) for im in self.settings["dataset"][dataset_name]]
        self.exposure_times = np.array([get_exposure(Image.open(im)) for im in self.settings["dataset"][dataset_name]], dtype= np.float32)
        self.tonemapAlgo = cv2.createTonemapDrago()
        self.result_merge = None
        self.result_img = None

    def applyDebevec(self):
        merge = cv2.createMergeDebevec()
        self.result_merge = merge.process(self.images, times= self.exposure_times.copy())
        
    def tonemap(self):
        self.result_img = np.clip(self.tonemapAlgo.process(self.result_merge.copy()) * 255, 0, 255).astype('uint8')

    def applyAceWindowed(self, image_index, window):
        self.result_img = windowed_ace.compute(self.images_paths[image_index], window)
    
    def applyAceExhaustive(self, image_index):
        self.result_img = exhaustive_ace.compute(self.images_paths[image_index])

    def save_image(self):
        if self.result_img is not None:
            cv2.imwrite(path.join(self.settings["save_path"], f"result_{self.dataset_name}.jpg"), self.result_img)


def main():
    hdr = HdrImplementations(dataset_name="star")
    # hdr.applyDebevec()
    # hdr.tonemap()
    # hdr.save_image()

if __name__ == "__main__":
    main()