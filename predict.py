import cv2
import argparse as ap
import numpy as np

from srcs.DetectionAgent import DetectionAgent


def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('--source', nargs='+', help='Paths to images to predict on (can be multiple)')
    parser.add_argument('--model', default='DetectionAgent_1', help='Model folder to load (default: DetectionAgent_1)')

    return parser.parse_args()


def image_to_unicode(img, width=40):
    height = int(width * img.shape[0] / img.shape[1] / 2)  # /2 car caractères hauts
    small_img = cv2.resize(img, (width, height))

    # Caractères de bloc Unicode
    blocks = [" ", "▀", "▄", "█"]

    result = ""
    for i in range(0, height, 2):
        for j in range(width):
            if i + 1 < height:
                # Prend deux pixels (haut et bas)
                top = small_img[i, j]
                bottom = small_img[i + 1, j] if i + 1 < height else [0, 0, 0]
                # Simplification : moyenne des canaux
                top_val = np.mean(top) // 64
                bottom_val = np.mean(bottom) // 64
                # Index dans les blocs
                idx = (top_val > 1) * 2 + (bottom_val > 1)
                result += blocks[idx]
            else:
                result += " "
        result += "\n"

    return result


if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        agent = DetectionAgent.load(args.model)

        for img_path in args.source:
            img = cv2.imread(img_path)
            prediction, transformed_images = agent.predict(img)

            print('\n====================================================\n')

            # Afficher l'image originale en ASCII
            print(image_to_unicode(img))

            # Afficher les images transformées
            for transformed_img in transformed_images:
                print(image_to_unicode(transformed_img))

            print('\n=====            DL classification            =====')
            print(f'\n           Class predicted : {prediction}')
            print('\n====================================================\n')

    except Exception as error:
        print(f'Error: {error}')
