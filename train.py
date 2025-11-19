import argparse as ap

from Augmentation import ImgAugmentation
from Transformation import ImgTransformator

from srcs.tools import load_original_images


def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-imgs_folder', default='data/leaves_test', help='original images folder')
    parser.add_argument('-save_file', default='training.zip', help='output file')
    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()
        original_images = load_original_images(args.imgs_folder)

        training_dataset = []

        augmentator = ImgAugmentation(original_images)
        transformator = ImgTransformator(original_images)

        augmented_imgs = augmentator.augment()
        transformed_imgs = transformator.transform()




    except Exception as error:
        print(f'Error: {error}')