import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage import affine_transform, map_coordinates, geometric_transform
from skimage.transform import resize, warp
from wand.image import Image
from wand.display import display
from colorama import Style, Fore

def DisplayImage(title, img):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def FlipImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    flipped_img = np.flipud(img)
    if action == 'display':
        DisplayImage('Flipped', flipped_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_flipped.JPG')
        plt.imsave(save_path, flipped_img)
        

def RotateImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    rotated_img = np.rot90(img, k=45)
    if action == 'display':
        DisplayImage('Rotated', rotated_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_rotated.JPG')
        plt.imsave(save_path, rotated_img)

def SkewImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    skew_x = 0.1
    skew_y = 0.1
    transform_matrix = np.array([[1, skew_x, 0],
                                [skew_y, 1, 0],
                                [0, 0, 1]])
    skewed_img = affine_transform(img, transform_matrix)
    if action == 'display':
        DisplayImage('Skewed', skewed_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_skewed.JPG')
        plt.imsave(save_path, skewed_img)

def ShearImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    shear_x = 0.4
    shear_matrix_horizontal = np.array([[1, shear_x, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
    sheared_img = affine_transform(img, shear_matrix_horizontal)
    if action == 'display':
        DisplayImage('Sheared', sheared_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_sheared.JPG')
        plt.imsave(save_path, sheared_img)

def CropImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    crop_height = np.random.randint(img.shape[0] // 2, img.shape[0])
    crop_width = np.random.randint(img.shape[1] // 2, img.shape[1])
    top = np.random.randint(0, img.shape[0] - crop_height)
    left = np.random.randint(0, img.shape[1] - crop_width)
    crop = img[top:top+crop_height, left:left+crop_width, :]
    resized_img = resize(crop, (224, 224))
    if action == 'display':
        DisplayImage('Cropped', resized_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_resized.JPG')
        plt.imsave(save_path, resized_img)

def DistortImage(img_path, action='display'):
    img = mpimg.imread(img_path)
    transform_matrix = np.array([[1, 0.2, 0],
                                 [0.2, 1, 0],
                                 [0, 0, 1]])
    distorted_img = affine_transform(img, transform_matrix)
    if action == 'display':
        DisplayImage('Distorted', distorted_img)
    elif action == 'save':
        save_path = img_path.replace('.JPG', '_distorted.JPG')
        plt.imsave(save_path, distorted_img)


def DisplayAugmentation():
    img_path = sys.argv[1]
    img = mpimg.imread(img_path)
    augmentations = [
        FlipImage,
        RotateImage,
        SkewImage,
        ShearImage,
        CropImage,
        DistortImage
    ]
    DisplayImage('Original', img)

    for func in augmentations:
        func(img_path)


def RepoAugmentation(repo_path):
    sub_repos = os.listdir(repo_path)
    for repo in sub_repos:
        repo_content = os.listdir(repo_path + '/' + repo)
        for img in repo_content:
            img_path = repo_path + '/' + repo + '/' + img
            FlipImage(img_path, action='save')
            RotateImage(img_path, action='save')
            SkewImage(img_path, action='save')
            ShearImage(img_path, action='save')
            CropImage(img_path, action='save')
            DistortImage(img_path, action='save')


if __name__ == '__main__':
    try: 
        if len(sys.argv) > 1:
            DisplayAugmentation()
        else:
            repo_path = input(Fore.GREEN + 'Path to repo ==> ' + Style.RESET_ALL)
            RepoAugmentation(repo_path)

    except Exception as error:
        print (error)