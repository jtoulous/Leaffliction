import os
import warnings
import argparse as ap
import numpy as np

from srcs.tools import load_original_images
from srcs.DetectionAgent import DetectionAgent


# Suppression des warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-imgs_folder', default='data/leaves', help='original images folder')
    parser.add_argument('-save_folder', default='DetectionAgent_1', help='output file')
    parser.add_argument('-epochs', type=int, default=10, help='')
    parser.add_argument('-transfo', default=['gaussian_blur'], nargs='+', help='transfo')

    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        X = []
        y = []

        total = 0

        original_images, type_of_load = load_original_images(args.imgs_folder)
        for img_class, imgs_list in original_images.items():
            for img_name, img_types in imgs_list.items():
                for img_type, img in img_types.items():

                    if not isinstance(img, np.ndarray):
                        print(f"ERREUR: {img_class}/{img_name} n'est pas un numpy array, type: {type(img)}")
                        continue

                    # Vérifiez les dimensions
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        print(f"ATTENTION: {img_class}/{img_name} a des dimensions invalides: {img.shape}")
                        continue

                    X.append(img)
                    y.append(img_class)

        X = np.array(X)
        y = np.array(y)

        agent = DetectionAgent(epochs=args.epochs, transfo=args.transfo)
        agent.train(X, y, transfo=args.transfo)
        agent.save(args.save_folder)


    except Exception as error:
        print(f'Error: {error}')