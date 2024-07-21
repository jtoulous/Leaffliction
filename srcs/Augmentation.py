import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import affine_transform, map_coordinates, geometric_transform
from skimage.transform import resize, warp
from wand.image import Image
from wand.display import display
from PIL import Image as PILImage
import io
from colorama import Style, Fore

def FlipImage(img):
    return np.flipud(img)

def RotateImage(img):
    return np.rot90(img, k=45)

def SkewImage(img):
    skew_x = 0.1
    skew_y = 0.1
    transform_matrix = np.array([[1, skew_x, 0],
                                [skew_y, 1, 0],
                                [0, 0, 1]])
    return affine_transform(img, transform_matrix)

def ShearImage(img):
    shear_x = 0.2
    shear_matrix_horizontal = np.array([[1, shear_x, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    return affine_transform(img, shear_matrix_horizontal)

def CropImage(img):
    crop_height = np.random.randint(img.shape[0] // 2, img.shape[0])
    crop_width = np.random.randint(img.shape[1] // 2, img.shape[1])
    top = np.random.randint(0, img.shape[0] - crop_height)
    left = np.random.randint(0, img.shape[1] - crop_width)
    crop = img[top:top+crop_height, left:left+crop_width, :]
    return resize(crop, (224, 224))

#def DistortImage(img_path):


def DisplayTransformations():
    img_path = sys.argv[1]
    img = mpimg.imread(sys.argv[1])
    augmentations = {
            'Flip': FlipImage,
            'Rotate': RotateImage,
            'Skew': SkewImage,
            'Shear': ShearImage,
            'Crop': CropImage,
#            'Distort': DistortImage
        }

    plt.imshow(img)
    plt.title('Original')
    plt.show()    

    for key, func in augmentations.items():
        tmp_img = func(img)
        plt.imshow(tmp_img)
        plt.title(key)
        plt.show()



#def RepoTransformation(repo_path)


if __name__ == '__main__':
    try: 
        if len(sys.argv) > 1:
            DisplayTransformations()
#        else:
#            repo_path = input(Fore.GREEN + 'Path to repo ==> ' + Style.RESET_ALL)
#            RepoTransformation(repo_path)

    except Exception as error:
        print (error)