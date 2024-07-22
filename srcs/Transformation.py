import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plantcv import plantcv as pcv
from colorama import Style, Fore

def DisplayImage(title, img):
    plt.imshow(img)
    plt.title(title)
    plt.show()

def GaussianBlur(img_path, action='display'):
    img = mpimg.imread(img_path)
    blurred_img = pcv.gaussian_blur(img=img, ksize=(51, 51), sigma_x=0, sigma_y=None)
    if action == 'display':
        DisplayImage('blurred', blurred_img)
    elif action == 'save':
        save_path = img_path.replace('.PNG', '_blurred.PNG')
        plt.imsave(save_path, blurred_img)

#def Mask(img_path, action='display'):
#    img = mpimg.imread(img_path)

#def RoiObjects(img_path, action='display'):

#def AnalyzeObject(img_path, action='display')

#def AnalyzeObject(img_path, action='display'):

#def Pseudolandmarks(img_path, action='display'):

def DisplayTransformation():
    img_path = sys.argv[1]
#    img = mpimg.imread(img_path)
    transformations = [
        GaussianBlur,
#        Mask,
#        RoiObjects,
#        AnalyzeObject,
#        Pseudolandmarks,
    ]

    for func in transformations:
        func(img_path)



if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            DisplayTransformation()
#        else:
#            repo_path = input(Fore.GREEN + 'Path to repo ==> ' + Style.RESET_ALL)
#            RepoTransformation()

    except Exception as error:
        print(error)