import os
import sys

import numpy as np
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
    img = mpimg.imread(img_path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img

    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    white_background = np.full_like(img_rgb, 255)
    masked_image = cv2.bitwise_or(img_rgb, img_rgb, mask=mask)
    white_background[mask_inv == 255] = masked_image[mask_inv == 255]
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
#    mask = mask.astype(np.uint8)
#    masked_image = pcv.apply_mask(img=img_rgb, mask=mask, mask_color='white')
    if action == 'display':
        return masked_image
#    elif action == 'save':


#def RoiObjects(img_path, action='display'):
#    img = mpimg.imread(img_path)
#    if img.dtype != np.uint8:
#        img = (img * 255).astype(np.uint8)
#    
#    if len(img.shape) == 2:
#        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#    elif img.shape[2] == 4:
#        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#    else:
#        img_rgb = img
#    
#    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#    _, mask = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
#    
#    masked = pcv.apply_mask(img_rgb, mask, 'black')
#    
#    # Calculer les dimensions de la ROI en fonction de la taille de l'image
#    height, width = img_rgb.shape[:2]
#    roi_width = int(width * 0.8)  # 80% de la largeur de l'image
#    roi_height = int(height * 0.8)  # 80% de la hauteur de l'image
#    roi_x = int((width - roi_width) / 2)  # Centré horizontalement
#    roi_y = int((height - roi_height) / 2)  # Centré verticalement
#    
#    roi = pcv.roi.rectangle(img=masked, x=roi_x, y=roi_y, h=roi_height, w=roi_width)
#    
#    roi_objects, roi_hierarchy = pcv.find_objects(img=masked, mask=mask)
#    roi_filtered = pcv.roi_objects(img=img_rgb, roi_contour=roi, roi_hierarchy=roi_hierarchy,
#                                   object_contour=roi_objects, obj_hierarchy=roi_hierarchy)
#    
#    img_copy = img_rgb.copy()
#    for obj in roi_filtered:
#        cv2.drawContours(img_copy, [obj], -1, (255, 0, 0), 2)
#    
#    if action == 'display':
#        return img_copy        


#def AnalyzeObject(img_path, action='display')

#def Pseudolandmarks(img_path, action='display'):


def DisplayTransformation():
    img_path = sys.argv[1]
    original_img = mpimg.imread(img_path)
    img_list = [original_img]
    transformations = [
        GrayScale,
#        Mask,
#        RoiObjects
    ]
    
    for func in transformations:
        transformed_img = func(img_path)
        if transformed_img.dtype != np.uint8:
            transformed_img = (transformed_img * 255).astype(np.uint8)
        if len(transformed_img.shape) == 2:
            transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_GRAY2RGB)
        img_list.append(transformed_img)

    DisplayImages(img_list, ["Original", "Gray scaled"])
#    DisplayImages(img_list, ["Original", "Gray scaled", "Masked", "Roi objects"])


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            DisplayTransformation()
#        else:
#            repo_path = input(Fore.GREEN + 'Path to repo ==> ' + Style.RESET_ALL)
#            RepoTransformation()

    except Exception as error:
        print(error)

