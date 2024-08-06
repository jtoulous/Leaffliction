import sys
from plantcv import plantcv as pcv
import numpy as np

def increase_brightness(image, mask, value):
    image = image.astype(np.uint8)
    bright_img = image.copy()
    bright_img[mask > 0] = np.clip(bright_img[mask > 0] + value, 0, 255)
    
    return bright_img


if __name__ == '__main__':
    try:
        img, _, _ = pcv.readimage(sys.argv[1])
        
        gray_img = pcv.rgb2gray_lab(img, 'l')
        pcv.plot_image(gray_img)

        gaussian_img = pcv.gaussian_blur(img=gray_img, ksize=(3, 3), sigma_x=0, sigma_y=None)
        pcv.plot_image(gaussian_img)

        threshold_img = pcv.threshold.gaussian(gray_img=gaussian_img, ksize=1000, offset=20, object_type='dark')
        pcv.plot_image(threshold_img)

        er_img = pcv.erode(gray_img=threshold_img, ksize=2, i=1)
        pcv.plot_image(er_img)

        dil_img = pcv.dilate(gray_img=er_img, ksize=4, i=1)
        pcv.plot_image(dil_img)

        masked_img = pcv.apply_mask(gray_img, dil_img, 'black')
        pcv.plot_image(masked_img)
    
        bright_masked_img = increase_brightness(masked_img, dil_img, value=100)
        pcv.plot_image(bright_masked_img, title="Brightened Masked Image")

    except Exception as error:
        print(error)