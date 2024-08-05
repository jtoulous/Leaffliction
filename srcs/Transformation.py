import sys
import os
import argparse
import cv2

from plantcv import plantcv as pcv
from utils.tools import LoadImage, ShowImage, DisplayImages

def Parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source repository/image')
    parser.add_argument('-dst', type=str, help='destination repository')
    return parser.parse_args()


def GrayScale(img, action="display"):
    scaled_img = pcv.rgb2gray_lab(img, 'l')
    
    binary_img = pcv.threshold.otsu(scaled_img, 'light')
#    ShowImage(binary_img)
    binary_img = pcv.invert(binary_img)
#    ShowImage(binary_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#    binary_img = cv2.erode(binary_img, kernel, iterations=1)
#    ShowImage(binary_img)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)
#    ShowImage(binary_img)
#    binary_img = pcv.fill(binary_img, 100)
#    ShowImage(binary_img)
    masked_img = pcv.apply_mask(scaled_img, binary_img, 'black')

    if action == "display":
        return masked_img
#    elif action == "save":


#def Mask(img, action='display'):
    


def DisplayTransformations(img_path):
    original_img = LoadImage(img_path)
    img_list = [original_img]
    transformations = [
        GrayScale,
#        Mask,
#        RoiObjects,
    ]

    for func in transformations:
        transformed_img = func(original_img)
        img_list.append(transformed_img)

    DisplayImages(img_list, ["Original", "Gray scale"])



if __name__ == '__main__':
    try:
        args = Parsing()
        if os.path.isfile(args.src):
            DisplayTransformations(args.src)
#        elif os.path.isdir(args.src):
#            RepoTransformations(args.src, args.dst)
        else:
            raise Exception('Error: the source file type is incompatible')

    except Exception as error:
        print(error)