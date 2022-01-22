import json
import os.path as path

import cv2
import numpy as np
from PIL import Image

import exhaustive_ace
import windowed_ace
import debevec
import gradient
import exposure_fusion
from utils import get_exposure

# create folder if it does not exists 
# save inside it
# Create a setup file?



class HdrImplementations():

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

        self.settings = json.load(open("settings.json"))
        self.images_paths = self.settings["dataset"][dataset_name]
        self.images = [cv2.imread(im) for im in self.settings["dataset"][dataset_name]]
        
        # just to make things faster
        # for i in range(len(self.images)):
        #   self.images[i] = cv2.resize(self.images[i], (1000, 1000), interpolation= cv2.INTER_AREA)

        self.exposure_times = [get_exposure(Image.open(im)) for im in self.settings["dataset"][dataset_name]]
       
        #self.exposure_times = [get_exposure(Image.open(im)) for im in self.settings["dataset"][dataset_name]]
        # self.exposure_times = [0.0125, 0.125, 0.5]
        self.tonemapAlgo = cv2.createTonemapDrago(1.0, 0.7)
        self.result_merge = None
        self.result_img = None

    def applyDebevecArt(self): #debevec python
        merge = cv2.createMergeDebevec()
        self.result_merge = merge.process(self.images, times= self.exposure_times.copy())
        
    
    def __tonemap(self):
        # self.result_img = np.clip(self.tonemapAlgo.process(self.result_merge.copy()) * 255, 0, 255).astype('uint8')
        self.result_img = self.tonemapAlgo.process(self.result_merge)
        self.result_img = 3 * self.result_img
        self.result_img = self.result_img * 255
        self.result_img = np.clip(self.result_img, 0, 255).astype('uint8')
        
       

    def applyAceWindowed(self, image_index, window):
        self.result_img = windowed_ace.compute(self.images_paths[image_index], window)
        # self.tonemap()
    
    def applyAceExhaustive(self, image_index):
    
        self.result_img = exhaustive_ace.compute(self.images_paths[image_index])
        # self.tonemapAlgo = cv2.createTonemapReinhard()
        # self.result_img = np.clip(self.tonemapAlgo.process(self.result_img.copy()) * 255, 0, 255).astype('uint8')
        # self.__tonemap()

    def applyDebevec(self): #seba python
        self.result_merge = debevec.compute(self.images, self.exposure_times)
        self.__tonemap()

    def applyGradient(self):
        self.result_img = gradient.compute(self.images)
        
    def applyExpFusion(self):
        self.result_img = exposure_fusion.compute(self.images)

    def save_image(self, name):
        if self.result_img is not None:
            cv2.imwrite(path.join(self.settings["save_path"], name), self.result_img)


def main():

    hdr = HdrImplementations(dataset_name="star")
    # hdr.applyAceWindowed(1, 50)
    # hdr.applyGradient()
    # hdr.applyExpFusion()
    # hdr.applyDebevec()
    # hdr.__tonemap()
    # hdr.save_image("ExpFusionUnderOver.jpg")
    # hdr.applyDebevec()
    

if __name__ == "__main__":
    main()
