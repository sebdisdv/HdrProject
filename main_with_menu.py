import json
import os.path as path
from telnetlib import EXOPL

import cv2
from cv2 import Algorithm
import numpy as np
from PIL import Image

import exhaustive_ace
import windowed_ace
import debevec
import gradient
import exposure_fusion
from utils import get_exposure

from consolemenu import SelectionMenu

DATASETS = ["Legno", "Stella", "Alberi", "Disco"]
IMGLIST = ["Under", "Mid", "Over"]
ALGORITHMS = ["ACE", "ACE_Windowed", "Debevec", "Mertens"]

def select_algorithm():
    algorithm_selection = SelectionMenu(ALGORITHMS, "Select which algorithm to use", show_exit_option=False)

    algorithm_selection.show()

    algorithm_selection.join()

    return algorithm_selection.selected_option

def select_dataset():
    dataset_selection = SelectionMenu(DATASETS, "Select a dataset to use", show_exit_option=False)

    dataset_selection.show()

    dataset_selection.join()

    return dataset_selection.selected_option

def select_image():
    img_selection = SelectionMenu(IMGLIST, "Select image to use", show_exit_option=False)

    img_selection.show()

    img_selection.join()

    return img_selection.selected_option

def select_quit():
    quit = SelectionMenu(["No", "Yes"], "Continue with another image | algorithm ?", show_exit_option=False, clear_screen=False)
    quit.show()
    quit.exit()

    return quit.selected_option - 1 

class HdrImplementations():

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

        self.settings = json.load(open("settings.json"))
        self.images_paths = self.settings["dataset"][dataset_name]
        self.images = [cv2.imread(im) for im in self.settings["dataset"][dataset_name]]

        self.exposure_times = [get_exposure(Image.open(im)) for im in self.settings["dataset"][dataset_name]]
       
        #self.exposure_times = [get_exposure(Image.open(im)) for im in self.settings["dataset"][dataset_name]]
        #self.exposure_times = [0.0125, 0.125, 0.5]
        self.tonemapAlgo = cv2.createTonemapDrago(1.0, 0.7)
        self.result_merge = None
        self.result_img = None

    def applyDebevecArt(self): #debevec python
        merge = cv2.createMergeDebevec()
        self.result_merge = merge.process(self.images, times= self.exposure_times.copy())
        
    
    def __tonemap(self):
        # self.result_img = np.clip(self.tonemapAlgo.process(self.result_merge.copy()) * 255, 0, 255).astype('uint8')
        self.result_img = self.tonemapAlgo.process(self.result_merge)
        # self.result_img = 3 * self.result_img
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
            cv2.imwrite(path.join(self.settings["save_path"], self.dataset_name, f"{name}.jpg"), self.result_img)





def main():
    while True:
        algo_index = select_algorithm()
        dataset_index = select_dataset()
        img_index = -1
        if algo_index <= 1:
            img_index = select_image()

        hdr = HdrImplementations(dataset_name= DATASETS[dataset_index])
        print(f"Algorithm selected {ALGORITHMS[algo_index]}")
        print(f"Dataset selected {DATASETS[dataset_index]}")
        name_res = input("Insert name for the resulting image: ")

        if ALGORITHMS[algo_index] == "ACE":
            hdr.applyAceExhaustive(img_index)
        elif ALGORITHMS[algo_index] == "ACE_Windowed":
            window = int(input("Insert window size in the range 100 <= w <= 250: "))
            window = np.clip(window, 100, 250)
            hdr.applyAceWindowed(img_index, window)
        elif ALGORITHMS[algo_index] == "Debevec":
            hdr.applyDebevec()
        elif ALGORITHMS[algo_index] == "Mertens":
            hdr.applyExpFusion()

        hdr.save_image(name_res)

        print(f"Image has benn saved in Results/{DATASETS[dataset_index]}")

        if select_quit():
            exit()

if __name__ == "__main__":
    main()