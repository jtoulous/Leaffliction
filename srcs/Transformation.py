import os
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plantcv import plantcv as pcv
from colorama import Style, Fore
from utils.tools import DisplayImages

def GrayScale(img_path, action='display'):
    img = mpimg.imread(img_path)
    scaled_img = pcv.rgb2gray(rgb_img=img)
    if action == 'display':
        return scaled_img
#    elif action == 'save':

def Mask(img_path, action='display'):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    masked_image = pcv.apply_mask(img=img_rgb, mask=mask, mask_color='white')
    if 'action' == 'display':
        return masked_image
#    elif action == 'save':

#def RoiObjects(img_path, action='display'):

#def AnalyzeObject(img_path, action='display')

#def Pseudolandmarks(img_path, action='display'):

def DisplayTransformation():
    img_path = sys.argv[1]
    img_list = [mpimg.imread(img_path)]
    transformations = [
        GrayScale,
        Mask,
#        RoiObjects,
#        AnalyzeObject,
#        Pseudolandmarks,
    ]

    for func in transformations:
        img_list.append(func(img_path))
    DisplayImages(img_list, ["Original", "Gray scaled", "Masked"])


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            DisplayTransformation()
#        else:
#            repo_path = input(Fore.GREEN + 'Path to repo ==> ' + Style.RESET_ALL)
#            RepoTransformation()

    except Exception as error:
        print(error)