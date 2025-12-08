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
    parser.add_argument('--source', default='data/leaves', help='Folder with images to train on (default: data/leaves)')
    parser.add_argument('--destination', default='DetectionAgent_1', help='Model save folder (default: DetectionAgent_1)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--transfo', default=[], nargs='+', help='Transformations to apply during training')

    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        X = []
        y = []

        total = 0

        original_images, type_of_load = load_original_images(args.source)
        for img_class, imgs_list in original_images.items():
            for img_name, img_types in imgs_list.items():
                for img_type, img in img_types.items():
                    X.append(img)
                    y.append(img_class)

        X = np.array(X)
        y = np.array(y)

        agent = DetectionAgent(epochs=args.epochs, transfo=args.transfo)
        agent.train(X, y)
        agent.save(args.destination)


    except Exception as error:
        print(f'Error: {error}')
