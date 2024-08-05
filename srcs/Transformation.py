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
    scaled_img = pcv.rgb2gray(rgb_img=img)
    if action == "display":
        return scaled_img
#    elif action == "save":


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