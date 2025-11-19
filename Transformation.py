import cv2
import numpy as np
import argparse as ap
# import plantcv as pcv

from plantcv import plantcv as pcv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from srcs.tools import load_original_images, save_images


################################################################
################################################################
#####                Transformator Class                   #####

class ImgTransformator:
    def __init__(self, images_structure):
        self.images_structure = images_structure

        return

    def transform(self, image=None, progress=None, task=None, display=False, transform=None):
        function_map = {
            'gaussian_blur': self.gaussian_blur,
            'mask': self.mask,
            'roi_objects': self.roi_objects,
            'analyze_object': self.analyze_object,
            'pseudolandmarks': self.pseudolandmarks,
        }

        if image:
            transformed_images = {}
            transformed_images['original'] = image
            transformed_images[transform] = function_map[transform](image) if transform in function_map else None

            if display:
                for idx, img in enumerate(transformed_images.values()):
                    cv2.imshow(f"Transformed Image {idx+1}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return transformed_images
        else:
            if task is not None:
                progress.update(task, total=sum(len(imgs) for imgs in self.images_structure.values()))

            for category in self.images_structure:

                if task is not None:
                    progress.update(task, description=f"Images transformation: {category}")

                for img_key in self.images_structure[category]:
                    image = self.images_structure[category][img_key]
                    self.images_structure[category][img_key] = {
                        'original': image,
                        transform: function_map[transform](image) if transform in function_map else None,
                    }

                    if task is not None:
                        progress.update(task, advance=1)

                    if display:
                        for idx, img in enumerate(self.images_structure[category][img_key].values()):
                            cv2.imshow(f"Transformed Image {idx+1}", img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                if task is not None:
                    progress.update(task, description="↪ Images transformation")

        return self.images_structure


    def gaussian_blur(self, img, k_size=(3, 3)):
        # Convertir en HSV pour mieux segmenter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # mask vert
        lower_green = np.array([20, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # mask marron
        lower_brown = np.array([5, 30, 10])
        upper_brown = np.array([25, 220, 180])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # combiner les deux mask
        mask = cv2.bitwise_or(mask_green, mask_brown)

        # Nettoyer le mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Trouver les contours de la feuilles
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer un nouveau mask rempli
        filled_mask = np.zeros_like(mask)
        if contours:
            # Prendre le plus grand contour (la feuille)
            largest_contour = max(contours, key=cv2.contourArea)
            # REMPLIR tout l'intérieur du contour
            cv2.fillPoly(filled_mask, [largest_contour], 255)

        # Utiliser le mask rempli au lieu de celui par couleur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(gray, gray, mask=filled_mask)

        # Égalisation d'histogramme pour renforcer les contrastes
        result_eq = cv2.equalizeHist(result)

        # Appliquer le blur
        blurred = cv2.GaussianBlur(result_eq, k_size, 0)

        # Convertir en BGR pour sauvegarde
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return blurred_bgr


    def mask(self, img):
        # Convertir en HSV pour mieux segmenter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # VERT TRES RESTRICTIF (seulement vert foncé)
        lower_green = np.array([25, 80, 30])    # saturation haute, value basse
        upper_green = np.array([85, 255, 100])  # value max basse

        # MARRON TRES SOUPLE
        lower_brown = np.array([0, 20, 10])     # hue très large
        upper_brown = np.array([30, 255, 200])  # saturation/value larges

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # COMBINER
        leaf_mask = cv2.bitwise_or(mask_green, mask_brown)

        # RESULTAT
        result = np.ones_like(img) * 255
        result[leaf_mask > 0] = img[leaf_mask > 0]

        return result


    def roi_objects(self, img):
        # Define a region of interest (ROI) to isolate the leaf object
        h, w = img.shape[:2]

        # Create ROI as a rectangle covering the central area of the image
        roi_result = pcv.roi.rectangle(img=img, x=int(w*0.1), y=int(h*0.1),
                                       h=int(h*0.8), w=int(w*0.8))

        # Handle both single return value and tuple
        if isinstance(roi_result, tuple):
            roi_contour, roi_hierarchy = roi_result
        else:
            roi_contour = roi_result
            roi_hierarchy = None

        # Convert to grayscale and threshold to find leaf contours
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter objects within ROI
        filtered_contours = []
        for contour in contours:
            # Check if contour is within ROI bounds
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if (int(w*0.1) <= cx <= int(w*0.9) and
                    int(h*0.1) <= cy <= int(h*0.9)):
                    filtered_contours.append(contour)

        # Draw contours on image for visualization
        result = img.copy()
        if filtered_contours:
            cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
        if roi_contour is not None and isinstance(roi_contour, np.ndarray):
            cv2.rectangle(result, (int(w*0.1), int(h*0.1)),
                         (int(w*0.9), int(h*0.9)), (255, 0, 0), 2)

        return result


    def analyze_object(self, img):
        transformed_img = img.copy()
        return transformed_img


    def pseudolandmarks(self, img):
        transformed_img = img.copy()
        return transformed_img

#####                                                      #####
################################################################






def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--load-folder',
        type=str,
        default='data/leaves',
        help='Folder with original images (default: data/leaves)')
    parser.add_argument(
        '--save-folder',
        type=str,
        default='data/leaves_preprocessed',
        help='Folder to save augmented images \
              (default: data/leaves_preprocessed)')
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display augmented images during processing (default: False)')
    parser.add_argument(
        '--range-nb',
        type=int,
        default=None,
        help='Number of augmented images to process (default: None)')
    parser.add_argument(
        '--range-percent',
        type=int,
        default=100,
        help='Percentage of augmented images to process (default: 100)')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)')
    parser.add_argument(
        '--transform',
        type=str,
        choices=['gaussian_blur', 'mask', 'roi_objects', 'analyze_object', 'pseudolandmarks'],
        default=None,
        help='Transformation to apply to images (default: None)')

    return parser.parse_args()


def range_processing(images, range_nb=None, range_percent=100):
    all_images = [(cat, img_key, img) for cat, imgs in images.items() for img_key, img in imgs.items()]
    all_images = all_images[:range_nb] if range_nb is not None else all_images

    limit = int(len(all_images) * range_percent / 100)
    all_images = all_images[:limit]

    images = {}
    for cat, img_key, img in all_images:
        if cat not in images:
            images[cat] = {}
        images[cat][img_key] = img

    return images


if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.completed}/{task.total}",
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
        ) as progress:
            global_task = progress.add_task("Global Progress", total=3)

            # Load images
            images_load_task = progress.add_task("↪ Load images", total=0)
            images = load_original_images(args.load_folder, progress=progress, task=images_load_task)
            images = range_processing(images, range_nb=args.range_nb, range_percent=args.range_percent)
            progress.update(global_task, advance=1)

            np.random.seed(args.seed)

            # Transform images
            images_transform_task = progress.add_task("↪ Images Transformation", total=0)
            transformator = ImgTransformator(images)
            transformed_images = transformator.transform(progress=progress, task=images_transform_task, display=args.display, transform=args.transform)
            progress.update(global_task, advance=1)

            # Save transformed images
            if args.save_folder not in [None, '', 'None']:
                images_save_task = progress.add_task("↪ Save transformed images", total=0)
                save_images(transformed_images, args.save_folder, progress=progress, task=images_save_task)
            progress.update(global_task, advance=1)


    except Exception as error:
        print(f'Error: {error}')
