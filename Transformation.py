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
            'spots_isolation': self.spots_isolation,
        }

        if image:
            transformed_images = {}
            transformed_images['original'] = image
            if transform in function_map:
                transformed_images[transform] = function_map[transform](image)
            else:
                for trans_name, trans_function in function_map.items():
                    transformed_images[trans_name] = trans_function(image)

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
                        **(
                            {trans_name: trans_function(image) for trans_name, trans_function in function_map.items()}
                            if transform is None else
                            {transform: function_map[transform](image)}
                        )
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
        # Apply mask to isolate the leaf - EXCLUDE SHADOWS
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # BROADER color ranges to capture more leaf variations
        # Green: capture light to dark greens, but exclude very dark (shadows)
        lower_green = np.array([20, 40, 40])  # Increased Value to exclude dark shadows
        upper_green = np.array([90, 255, 255])

        # Brown/Yellow: capture diseased/autumn leaves, exclude dark shadows
        lower_brown = np.array([5, 40, 40])  # Increased Saturation and Value
        upper_brown = np.array([35, 255, 220])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        leaf_mask = cv2.bitwise_or(mask_green, mask_brown)

        # ADDITIONAL: Exclude very dark regions (shadows) using Value channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Get grayscale
        _, bright_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  # Exclude pixels darker than 50
        leaf_mask = cv2.bitwise_and(leaf_mask, bright_mask)  # Keep only bright regions

        # Clean up the mask with LARGER kernel for better filling
        kernel = np.ones((7, 7), np.uint8)
        # Close gaps first
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Fill holes
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        # Remove small noise
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_ERODE, kernel, iterations=1)

        # Find ALL contours and select the largest
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = img.copy()

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Optional: smooth the contour using convex hull or approxPolyDP
            # Uncomment one of these if contour is too jagged:
            # largest_contour = cv2.convexHull(largest_contour)
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Get bounding rectangle around the leaf
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2*padding)
            h = min(img.shape[0] - y, h + 2*padding)

            # Use PlantCV's ROI from the bounding box
            roi = pcv.roi.rectangle(img=result, x=x, y=y, h=h, w=w)

            # Draw the leaf contour in green (THICKER for visibility)
            cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

            # Draw the bounding rectangle in blue
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)

        return result


    def analyze_object(self, img):
        transformed_img = img.copy()
        return transformed_img


    def pseudolandmarks(self, img):
        transformed_img = img.copy()
        return transformed_img


    def spots_isolation(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Mask large pour trouver TOUS les marrons possibles
        mask_all_brown = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([40, 255, 220]))
        
        # Calculer l'histogramme des valeurs (V) des pixels marrons
        brown_pixels_v = hsv[:,:,2][mask_all_brown > 0]
        
        if len(brown_pixels_v) > 0:
            median_v = np.median(brown_pixels_v)
            v_range = 40
            lower_v = max(30, median_v - v_range)
            upper_v = min(200, median_v + v_range)
        else:
            lower_v, upper_v = 50, 150
        
        lower_brown = np.array([0, 30, lower_v])
        upper_brown = np.array([30, 220, upper_v])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # DILATATION POUR COMBLER LES TROUS ET ÉTENDRE LES ZONES
        kernel = np.ones((5,5), np.uint8)
        mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)  # Combine d'abord les zones proches
        mask_brown = cv2.dilate(mask_brown, kernel, iterations=1)  # Étend les bords
        
        result = np.ones_like(img) * 255
        result[mask_brown > 0] = img[mask_brown > 0]
    
        return result

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
        help='Number of images to process (default: None)')
    parser.add_argument(
        '--range-percent',
        type=int,
        default=100,
        help='Percentage of images to process (default: 100)')
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
    np.random.shuffle(all_images)
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
