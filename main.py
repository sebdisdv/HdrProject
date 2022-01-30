import exhaustive_ace
import windowed_ace
import debevec
import gradient
import exposure_fusion
import cv2
import numpy as np
import os.path as path

from PIL import Image
from utils import create_folders, get_exposure, get_dataset_info
from simple_term_menu import TerminalMenu
from termcolor import colored

ALGORITHMS = ["ACE", "ACE_Windowed", "Debevec", "Mertens"]
QUIT = ["No", "Yes"]


def select_algorithm():
    menu = TerminalMenu(
        ALGORITHMS, clear_screen=True, title="\nSelect which algorithm to use\n"
    )
    return ALGORITHMS[menu.show()]


def select_dataset(names):
    menu = TerminalMenu(names, title="\nSelect which Dataset to use\n")
    return names[menu.show()]


def select_image(imgs_names):
    menu = TerminalMenu(imgs_names, title="\nSelect which image to use\n")
    return menu.show()


def select_quit():
    menu = TerminalMenu(QUIT, title="\nDo you want to quit?\n")
    return QUIT[menu.show()]


class HdrImplementations:
    def __init__(self, dataset_name: str, imgs_names) -> None:
        self.dataset_name = dataset_name
        self.images_paths = [
            path.join("Dataset", dataset_name, img) for img in imgs_names
        ]
        self.images = [cv2.imread(im) for im in self.images_paths]

        self.exposure_times = [get_exposure(Image.open(im)) for im in self.images_paths]

        self.tonemapAlgo = cv2.createTonemapDrago(1.0, 0.7)
        self.result_merge = None
        self.result_img = None

    def applyDebevecArt(self):
        merge = cv2.createMergeDebevec()
        self.result_merge = merge.process(self.images, times=self.exposure_times.copy())

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
            cv2.imwrite(
                path.join("Results", self.dataset_name, f"{name}.jpg"),
                self.result_img,
            )


def main(names, info):
    while True:
        algorithm = select_algorithm()
        dataset = select_dataset(names)
        img_index = -1
        if algorithm in ["ACE", "ACE_Windowed"]:
            img_index = select_image(info[dataset])

        hdr = HdrImplementations(dataset_name=dataset, imgs_names=info[dataset])
        print(f"Algorithm selected {algorithm}")
        print(f"Dataset selected {dataset}")
        name_res = input("Insert name for the resulting image: ")

        if algorithm == "ACE":
            hdr.applyAceExhaustive(img_index)
        elif algorithm == "ACE_Windowed":
            window = int(input("Insert window size in the range 100 <= w <= 250: "))
            window = np.clip(window, 100, 250)
            hdr.applyAceWindowed(img_index, window)
        elif algorithm == "Debevec":
            hdr.applyDebevec()
        elif algorithm == "Mertens":
            hdr.applyExpFusion()

        hdr.save_image(name_res)

        print(colored(f"Image has benn saved in Results/{dataset}", "green"))

        if select_quit() == "Yes":
            exit()


if __name__ == "__main__":
    names, info = get_dataset_info()
    create_folders(names)
    main(names, info)
