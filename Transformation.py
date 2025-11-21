import cv2
import numpy as np
import argparse as ap

from plantcv.plantcv.homology.x_axis_pseudolandmark import x_axis_pseudolandmarks
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from srcs.tools import save_images, load_images


class ImgTransformator:
    def __init__(self, images_structure=None):
        """
        Initialize the ImgTransformator with a given structure of images.

        Args:
            images_structure (dict): A dictionary containing images categorized by class names.
        """
        self.images_structure = images_structure

        self.function_map = {
            'gaussian_blur': self.gaussian_blur,
            'mask': self.mask,
            'roi_objects': self.roi_objects,
            'pseudolandmarks': self.pseudolandmarks,
            'spots_isolation': self.spots_isolation,
            'background_removal': self.background_removal,
        }


    def transform(self, image=None, progress=None, task=None, display=False, transform=None, transform_list=None):
        """
        Apply transformations to images.

        Args:
            image (np.ndarray, optional): A single image to transform. If None, transforms all images in the structure.
            progress (Progress, optional): Rich Progress object for tracking progress.
            task (Task, optional): Rich Task object for updating progress.
            display (bool, optional): Whether to display images during transformation.
            transform (str, optional): Specific transformation to apply. If None, applies all transformations.

        Returns:
            dict: A dictionary containing transformed images.
        """
        function_map = {
            'gaussian_blur': self.gaussian_blur,
            'mask': self.mask,
            'roi_objects': self.roi_objects,
            'pseudolandmarks': self.pseudolandmarks,
            'spots_isolation': self.spots_isolation,
            'background_removal': self.background_removal,
        }

        if image is not None:
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
            if transform_list is not None:
                num_transformations = len(transform_list)
            elif transform is not None:
                num_transformations = 1
            else:
                num_transformations = len(function_map)

            total_operations = sum(len(imgs) for imgs in self.images_structure.values()) * (num_transformations + 1)

            if task is not None:
                progress.update(task, total=total_operations)

            for category in self.images_structure:

                if task is not None:
                    progress.update(task, description=f"↪ Images transformation: {category}")

                for img_key in self.images_structure[category]:
                    image = self.images_structure[category][img_key]['original']

                    self.images_structure[category][img_key] = {'original': image}
                    if task is not None:
                        progress.update(task, advance=1)

                    if transform is None and transform_list is None:
                        for trans_name, trans_function in function_map.items():
                            self.images_structure[category][img_key][trans_name] = trans_function(image)
                            if task is not None:
                                progress.update(task, advance=1)

                    elif transform_list is not None:
                        for transform in transform_list:
                            if transform in function_map:
                                self.images_structure[category][img_key][transform] = function_map[transform](image)
                                if task is not None:
                                    progress.update(task, advance=1)

                    else:
                        self.images_structure[category][img_key][transform] = function_map[transform](image)
                        if task is not None:
                            progress.update(task, advance=1)

                    if display:
                        transformed_images = self.images_structure[category][img_key]
                        for trans_type, trans_img in transformed_images.items():
                            cv2.imshow(f"{category} - {img_key} - {trans_type}", trans_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                if task is not None:
                    progress.update(task, description="↪ Images transformation")

        return self.images_structure

    def quick_use(self, img, function_name):
        transformed_image = self.function_map[function_name](img)
        return transformed_image



    def gaussian_blur(self, image, k_size=(3, 3)):
        """
        Apply Gaussian blur to the leaf area of the image.

        Args:
            image (np.ndarray): Input image.
            k_size (tuple, optional): Kernel size for Gaussian blur. Default is (3, 3).

        Returns:
            np.ndarray: Blurred image with leaf area enhanced.

        Behavior:
            - Segments the leaf using color thresholds in HSV space.
            - Creates a filled mask of the leaf area.
            - Applies histogram equalization to enhance contrast.
            - Applies Gaussian blur to the enhanced leaf area.
            - Returns the blurred image in BGR format.
        """
        transformed_image = image.copy()
        hsv = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array([20, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Brown mask
        lower_brown = np.array([5, 30, 10])
        upper_brown = np.array([25, 220, 180])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # Combine both masks
        mask = cv2.bitwise_or(mask_green, mask_brown)

        # Clean the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find leaf contours
        contours, _ = self._find_largest_contour(transformed_image)

        # Create a new filled mask
        filled_mask = np.zeros_like(mask)
        if contours:
            # Take the largest contour (the leaf)
            largest_contour = max(contours, key=cv2.contourArea)
            # Fill the entire inside of the contour
            cv2.fillPoly(filled_mask, [largest_contour], 255)

        # Use the filled mask instead of the color mask
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        transformed_image = cv2.bitwise_and(transformed_image, transformed_image, mask=filled_mask)

        # Histogram equalization to enhance contrast
        transformed_image = cv2.equalizeHist(transformed_image)

        # Apply Gaussian blur
        transformed_image = cv2.GaussianBlur(transformed_image, k_size, 0)

        # Convert to BGR for saving
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2BGR)

        return transformed_image

    def mask(self, image):
        """
        Apply a mask to isolate leaf areas in the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with non-leaf areas masked out.

        Behavior:
            - Finds the largest contour (leaf boundary) to remove background
            - Applies color-based mask within the leaf contour area
            - Returns the masked image with non-leaf areas set to white.
        """
        transformed_image = image.copy()
        contours, _ = self._find_largest_contour(transformed_image)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Create contour mask (removes background)
            contour_mask = np.zeros(transformed_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Convert to HSV for color-based segmentation
            hsv = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)

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

            # Apply contour mask to restrict color mask to leaf area only
            leaf_mask = cv2.bitwise_and(leaf_mask, contour_mask)

            # RESULTAT
            tmp = np.ones_like(transformed_image) * 255
            tmp[leaf_mask > 0] = transformed_image[leaf_mask > 0]

            transformed_image = tmp

        return transformed_image

    def roi_objects(self, image):
        """
        Draw the region of interest (ROI) around the leaf in the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with ROI drawn around the leaf.

        Behavior:
            - Finds the largest contour in the image (assumed to be the leaf).
            - Draws the contour in green and a bounding rectangle in blue.
            - Returns the image with the ROI highlighted.
        """
        transformed_image = image.copy()
        contours, _ = self._find_largest_contour(transformed_image)

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the leaf contour in green (THICKER for visibility)
            cv2.drawContours(transformed_image, [largest_contour], -1, (0, 255, 0), 2)

            # Get bounding rectangle around the leaf
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(transformed_image.shape[1] - x, w + 2*padding)
            h = min(transformed_image.shape[0] - y, h + 2*padding)

            # Draw the bounding rectangle in blue
            cv2.rectangle(transformed_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

        return transformed_image

    def pseudolandmarks(self, image):
        """
        Draw pseudolandmarks along the leaf's horizontal axis.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with pseudolandmarks drawn.

        Behavior:
            - Finds the largest contour in the image (assumed to be the leaf).
            - Uses PlantCV's x_axis_pseudolandmarks to compute pseudolandmarks.
            - Draws the top, bottom, and center vertical pseudolandmarks in different colors.
            - Returns the image with pseudolandmarks highlighted.
        """
        transformed_image = image.copy()
        contours, leaf_mask = self._find_largest_contour(transformed_image)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            cv2.drawContours(transformed_image, [largest_contour], -1, (0, 255, 0), 2)

            # Use PlantCV's x_axis_pseudolandmarks function (correct signature: img, mask)
            # This function generates pseudolandmarks along horizontal slices
            landmarks_result = x_axis_pseudolandmarks(img=transformed_image, mask=leaf_mask)

            # Check if the result is valid (not "NA")
            if not (isinstance(landmarks_result, np.ndarray) and landmarks_result.dtype.kind == 'U'):
                top, bottom, center_v = landmarks_result

                # Draw the pseudolandmarks
                # Top landmarks (upper leaf boundary)
                if top is not None and len(top) > 0:
                    for point in top:
                        # Convert to integer tuple (x, y)
                        pt = (int(point[0][0]), int(point[0][1]))
                        cv2.circle(transformed_image, pt, 4, (0, 0, 255), -1)  # Red - top boundary

                # Bottom landmarks (lower leaf boundary)
                if bottom is not None and len(bottom) > 0:
                    for point in bottom:
                        # Convert to integer tuple (x, y)
                        pt = (int(point[0][0]), int(point[0][1]))
                        cv2.circle(transformed_image, pt, 4, (255, 0, 0), -1)  # Blue - bottom boundary

                # Center vertical landmarks (leaf midline)
                if center_v is not None and len(center_v) > 0:
                    for point in center_v:
                        # Convert to integer tuple (x, y)
                        pt = (int(point[0][0]), int(point[0][1]))
                        cv2.circle(transformed_image, pt, 4, (255, 255, 0), -1)  # Green - centerline

        return transformed_image

    def spots_isolation(self, image):
        """
        Isolate spots (brown areas) on the leaf.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with isolated spots.

        Behavior:
            - Finds the largest contour (leaf boundary) to remove background
            - Converts the image to HSV color space.
            - Dynamically determines brown color thresholds based on the image.
            - Creates a mask for brown areas and applies morphological operations to enhance it.
            - Returns the image with only the brown areas visible within the leaf, non-brown areas set to white
        """
        transformed_image = image.copy()
        contours, _ = self._find_largest_contour(transformed_image)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Create contour mask (removes background)
            contour_mask = np.zeros(transformed_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            hsv = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV)

            # Mask large pour trouver TOUS les marrons possibles
            mask_all_brown = cv2.inRange(hsv, np.array([0, 20, 30], dtype=np.uint8), np.array([40, 255, 220], dtype=np.uint8))

            # Apply contour mask to restrict brown detection to leaf area only
            mask_all_brown = cv2.bitwise_and(mask_all_brown, contour_mask)

            # Calculer l'histogramme des valeurs (V) des pixels marrons
            brown_pixels_v = hsv[:, :, 2][mask_all_brown > 0]

            if len(brown_pixels_v) > 0:
                median_v = np.median(brown_pixels_v)
                v_range = 40
                lower_v = int(max(30, median_v - v_range))
                upper_v = int(min(200, median_v + v_range))
            else:
                lower_v, upper_v = 50, 150

            lower_brown = np.array([0, 30, lower_v], dtype=np.uint8)
            upper_brown = np.array([30, 220, upper_v], dtype=np.uint8)
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

            # Apply contour mask to restrict brown spots to leaf area only
            mask_brown = cv2.bitwise_and(mask_brown, contour_mask)

            # DILATATION POUR COMBLER LES TROUS ET ÉTENDRE LES ZONES
            kernel = np.ones((5, 5), np.uint8)
            mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)  # Combine d'abord les zones proches
            mask_brown = cv2.dilate(mask_brown, kernel, iterations=1)  # Étend les bords

            tmp = np.ones_like(transformed_image) * 255
            tmp[mask_brown > 0] = transformed_image[mask_brown > 0]

            transformed_image = tmp

        return transformed_image

    def background_removal(self, image):
        """
        Remove the background from the image, leaving only the leaf.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with background removed (white background).

        Behavior:
            - Finds the largest contour in the image (assumed to be the leaf).
            - Creates a precise mask from the contour.
            - Applies morphological operations and Gaussian blur to smooth edges.
            - Returns the image with the leaf on a white background.
        """
        transformed_image = image.copy()
        contours, _ = self._find_largest_contour(transformed_image)

        if contours:
            # Get the largest contour (the leaf)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a precise mask from the contour
            precise_mask = np.zeros(transformed_image.shape[:2], dtype=np.uint8)

            # Fill the contour to create a solid mask
            cv2.drawContours(precise_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Optional: Apply morphological operations to smooth the edges
            kernel = np.ones((3, 3), np.uint8)
            # Slightly erode to remove any background artifacts on edges
            precise_mask = cv2.erode(precise_mask, kernel, iterations=1)
            # Then dilate back to restore size but with smoother edges
            precise_mask = cv2.dilate(precise_mask, kernel, iterations=1)

            # Apply Gaussian blur to the mask for smoother edges (anti-aliasing)
            precise_mask = cv2.GaussianBlur(precise_mask, (5, 5), 0)

            # Create output with white background
            transformed_image = np.ones_like(transformed_image) * 255

            # Use the mask as alpha blending weight for smooth edges
            mask_3channel = cv2.cvtColor(precise_mask, cv2.COLOR_GRAY2BGR) / 255.0
            transformed_image = (image * mask_3channel + transformed_image * (1 - mask_3channel)).astype(np.uint8)

        return transformed_image

    def _find_largest_contour(self, image):
        """
        Find contours in the image and return them along with the leaf mask.

        Args:
            image (np.ndarray): Input image.

        Returns:
            tuple: (contours, leaf_mask) where contours is a list of detected contours and leaf_mask is the binary mask of the leaf area.

        Behavior:
            - Segments the leaf using color thresholds in HSV space.
            - Creates a binary mask of the leaf area, excluding shadows.
            - Cleans the mask using morphological operations.
            - Finds and returns all contours from the cleaned mask.
        """
        # Apply mask to isolate the leaf - EXCLUDE SHADOWS
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Get grayscale
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

        return contours, leaf_mask


def ArgumentParsing():
    """
    Parse command-line arguments for image augmentation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--source',
        type=str,
        default='data/leaves',
        help='Folder with original images (default: data/leaves)')
    parser.add_argument(
        '--destination',
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
        choices=['gaussian_blur', 'mask', 'roi_objects', 'pseudolandmarks', 'spots_isolation', 'background_removal'],
        default=None,
        help='Transformation to apply to images (default: None)')

    return parser.parse_args()


def range_processing(images, range_nb=None, range_percent=100):
    """
    Limit the number of images to process based on specified number and/or percentage.

    Args:
        images (dict): Dictionary of images categorized by class names.
        range_nb (int, optional): Maximum number of images to process. If None, no limit is applied.
        range_percent (int, optional): Percentage of images to process (0-100). Default is 100.

    Returns:
        dict: Dictionary of images limited to the specified number/percentage.

    Behavior:
        - Flattens the images dictionary into a list of (category, image_key, image) tuples.
        - Shuffles the list randomly.
        - Selects the first 'range_nb' images if specified.
        - Further limits the selection to 'range_percent' of the total images.
        - Reconstructs and returns a dictionary of the selected images.
    """
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
        np.random.seed(args.seed)

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
            images, type_of_load = load_images(args.source, progress=progress, task=images_load_task)
            images = range_processing(images, range_nb=args.range_nb, range_percent=args.range_percent)
            progress.update(global_task, advance=1)

            if type_of_load == 'File':
                args.display = True

            # Transform images
            images_transform_task = progress.add_task("↪ Images Transformation", total=0)
            transformator = ImgTransformator(images)
            transformed_images = transformator.transform(progress=progress, task=images_transform_task, display=args.display, transform=args.transform)
            progress.update(global_task, advance=1)

            # Save transformed images
            if args.destination not in [None, '', 'None']:
                images_save_task = progress.add_task("↪ Save transformed images", total=0)
                save_images(transformed_images, args.destination, progress=progress, task=images_save_task)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
