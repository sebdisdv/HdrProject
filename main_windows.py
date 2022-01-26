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
from utils import get_exposure, get_dataset_info, create_folders

from consolemenu import SelectionMenu
from termcolor import colored
from typing import List



IMGLIST = ["Under", "Mid", "Over"]
ALGORITHMS = ["ACE", "ACE_Windowed", "Debevec", "Mertens"]

def select_algorithm():
    algorithm_selection = SelectionMenu(ALGORITHMS, "Select which algorithm to use", show_exit_option=False, clear_screen=True)

    algorithm_selection.show()

    algorithm_selection.join()

    return algorithm_selection.selected_option

def select_dataset(names):
    dataset_selection = SelectionMenu(names, "Select which dataset to use", show_exit_option=False, clear_screen=False)

    dataset_selection.show()

    dataset_selection.join()

    return dataset_selection.selected_option

def select_image(names):
    img_selection = SelectionMenu(names, "Select which image to use", show_exit_option=False, clear_screen=False)

    img_selection.show()

    img_selection.join()

    return img_selection.selected_option

def select_quit():
    quit = SelectionMenu(["No", "Yes"], "Do you want to quit?", show_exit_option=False, clear_screen=False)
    quit.show()
    quit.exit()

    return quit.selected_option 

class HdrImplementations():

    def __init__(self, dataset_name: str, imgs_names: List[str]) -> None:
        self.dataset_name = dataset_name

        # self.settings = json.load(open("settings.json"))
        self.images_paths = [path.join("Dataset", dataset_name, img) for img in imgs_names]
        
        self.images = [cv2.imread(im) for im in self.images_paths]
        
        self.exposure_times = [get_exposure(Image.open(im)) for im in self.images_paths]
       
        #self.exposure_times = [0.0125, 0.125, 0.5]
        self.tonemapAlgo = cv2.createTonemapDrago(1.0, 0.7)
        self.result_merge = None
        self.result_img = None

    def applyDebevecArt(self): 
        merge = cv2.createMergeDebevec()
        self.result_merge = merge.process(self.images, times= self.exposure_times.copy())
               
    def applyAceWindowed(self, image_index, window):
        self.result_img = windowed_ace.compute(self.images_paths[image_index], window)
        
    
    def applyAceExhaustive(self, image_index):
        self.result_img = exhaustive_ace.compute(self.images_paths[image_index])
        

    def applyDebevec(self): 
        self.result_merge = debevec.compute(self.images, self.exposure_times)
        self.result_img = self.tonemapAlgo.process(self.result_merge.copy())
        if self.dataset_name == "Stella":
            self.result_img = 3 * self.result_img
        self.result_img = self.result_img * 255
        self.result_img = np.clip(self.result_img, 0, 255).astype("uint8")

    def applyGradient(self):
        self.result_img = gradient.compute(self.images)
        
    def applyExpFusion(self):
        self.result_img = exposure_fusion.compute(self.images)

    def save_image(self, name):
        if self.result_img is not None:
            cv2.imwrite(path.join("Results", self.dataset_name, f"{name}.jpg"), self.result_img)





def main(names, info):
    while True:

        algo_index = select_algorithm()
        dataset_index = select_dataset(names)
        img_index = -1
        
        if algo_index <= 1:
            img_index = select_image(info[names[dataset_index]])

        hdr = HdrImplementations(dataset_name= names[dataset_index], imgs_names= info[names[dataset_index]])
        print(f"Algorithm selected {ALGORITHMS[algo_index]}")
        print(f"Dataset selected {names[dataset_index]}")
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

        print(colored(f"\nImage has been saved in Results/{names[dataset_index]}", 'green'))

        if select_quit():
            exit()

if __name__ == "__main__":
    names, info = get_dataset_info()
    main(names, info)