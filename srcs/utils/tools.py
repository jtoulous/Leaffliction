import cv2
import matplotlib.pyplot as plt
import numpy as np

from plantcv import plantcv as pcv

def DisplayImages(img_list, title_list):
    for img, title in zip(img_list, title_list):
        ShowImage(img, title=title)

def LoadImage(img_path):
    img, _, _ = pcv.readimage(img_path)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def ShowImage(img, title=''):
    cv2.imshow(title, img)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.waitKey(100)
    cv2.destroyAllWindows()