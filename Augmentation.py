import cv2
import numpy as np
import argparse as ap

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from srcs.tools import load_original_images, save_images, range_processing


class ImgAugmentation:
    def __init__(self, images_structure):
        """
        Initialize the ImgAugmentation with a given structure of images.

        Args:
            images_structure (dict): A dictionary containing images categorized by class names.
        """
        self.images_structure = images_structure
        self.max_disease_count = max(len(imgs) for imgs in images_structure.values())

        return

    def update_image_struct(self, progress=None, task=None):
        """
        Update the images structure with oversampling.

        Args:
            progress (Progress, optional): Rich Progress object for tracking progress.
            task (Task, optional): Rich Task object for updating progress.
            display (bool, optional): Whether to display images during augmentation.
            augmentation (str, optional): Specific augmentation to apply. If None, applies all augmentations.

        Returns:
            dict: The updated images structure with oversampling.
        """
        augmentation_types = ['rotation', 'blur', 'contrast', 'scaling', 'illumination', 'projective']

        if task is not None:
            total_operations = sum(len(self.images_structure[category]) for category in self.images_structure)
            progress.update(task, total=total_operations)

        final_struct = {}

        for category in self.images_structure:
            final_struct[category] = {}
            current_count = len(self.images_structure[category])

            # Calculate how many transformations needed per image to reach max_disease_count
            total_needed = self.max_disease_count
            transformations_per_image = (total_needed // current_count) + 1

            for img_key, img in self.images_structure[category].items():
                final_struct[category][img_key] = {}

                # Always add original
                final_struct[category][img_key]['original'] = self.images_structure[category][img_key]['original']

                # Add augmentations randomly until we reach the needed count
                for i in range(transformations_per_image - 1):  # -1 because we already have 'original'
                    aug_type = np.random.choice(augmentation_types)

                    # Make unique keys if the same augmentation is used multiple times
                    aug_key = aug_type
                    counter = 1
                    while aug_key in final_struct[category][img_key]:
                        aug_key = f"{aug_type}_{counter}"
                        counter += 1

                    final_struct[category][img_key][aug_key] = 'TODO'

                if task is not None:
                    progress.update(task, advance=1)

            # Trim excess if we overshot the target
            total_transformations = sum(len(transforms) for transforms in final_struct[category].values())
            if total_transformations > total_needed:
                # Remove excess transformations from the last images
                excess = total_transformations - total_needed
                for img_key in list(final_struct[category].keys()):
                    if excess == 0:
                        break
                    transforms = list(final_struct[category][img_key].keys())
                    # Remove non-original transformations
                    for trans_key in reversed(transforms):
                        if trans_key != 'original' and excess > 0:
                            del final_struct[category][img_key][trans_key]
                            excess -= 1

        self.images_structure = final_struct

        return self.images_structure

    def augment(self, image=None, progress=None, task=None, display=False, augmentation=None):
        """
        Apply augmentations to images.

        Args:
            image (np.ndarray, optional): A single image to augment. If None, augments all images in the structure.
            progress (Progress, optional): Rich Progress object for tracking progress.
            task (Task, optional): Rich Task object for updating progress.
            display (bool, optional): Whether to display images during augmentation.
            augmentation (str, optional): Specific augmentation to apply. If None, applies all augmentations.

        Returns:
            dict: A dictionary containing augmented images.
        """
        function_map = {
            'rotation': self.rotation,
            'blur': self.blur,
            'contrast': self.contrast,
            'scaling': self.scaling,
            'illumination': self.illumination,
            'projective': self.projective,
        }

        if image:
            augmented_images = {}
            augmented_images['original'] = image
            if augmentation in function_map:
                augmented_images[augmentation] = function_map[augmentation](image)
            else:
                for aug_name, aug_function in function_map.items():
                    augmented_images[aug_name] = aug_function(image)

            if display:
                for idx, img in enumerate(augmented_images.values()):
                    cv2.imshow(f"Augmented Image {idx+1}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return augmented_images
        else:
            if task is not None:
                total = sum(len(imgs) for imgs in self.images_structure.values() for imgs in imgs.values())
                progress.update(task, total=total)

            for category in self.images_structure:
                for img_key in self.images_structure[category]:
                    augmented_images = {}
                    for aug_type in self.images_structure[category][img_key]:
                        if aug_type == 'original':
                            augmented_images['original'] = self.images_structure[category][img_key]['original']
                        else:
                            if augmentation is None or aug_type == augmentation:
                                aug_function = function_map.get(aug_type.split('_')[0], None)
                                if aug_function:
                                    augmented_images[aug_type] = aug_function(self.images_structure[category][img_key]['original'])
                        if task is not None:
                            progress.update(task, advance=1)
                    self.images_structure[category][img_key] = augmented_images

                    if display:
                        augmented_images = self.images_structure[category][img_key]
                        for aug_type, aug_image in augmented_images.items():
                            cv2.imshow(f"{category} - {img_key} - {aug_type}", aug_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

        return self.images_structure

    def rotation(self, image, angle=np.random.randint(20, 340)):
        """
        Apply rotation augmentation to the image.

        Args:
            image (np.ndarray): The input image to be rotated.
            angle (float, optional): The rotation angle in degrees. Defaults to a random angle between 20 and 340.

        Returns:
            np.ndarray: The rotated image.

        Behavior:
            - Computes new image bounds to ensure the entire rotated image fits.
            - Creates a rotation matrix centered on the image.
            - Adjusts translation to center the rotated image.
            - Applies the rotation using cv2.warpAffine with a white border.
            - Returns the rotated image.
        """
        if image is None:
            return []

        h, w = image.shape[:2]

        # compute new image bounds so the whole rotated image fits
        rad = np.deg2rad(angle)
        cos = abs(np.cos(rad))
        sin = abs(np.sin(rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # rotation matrix around the center of the image
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

        # adjust translation to put the rotated image in the center
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        # choose white border value
        # (single value for grayscale, tuple for color)
        if image.ndim == 2:
            border_value = 255
        else:
            border_value = tuple([255] * image.shape[2])

        rotated_image = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        return rotated_image

    def blur(self, image, blur_factor=5):
        """
        Apply Gaussian blur augmentation to the image.

        Args:
            image (np.ndarray): The input image to be blurred.
            blur_factor (int, optional): The size of the Gaussian kernel. Defaults to 15.

        Returns:
            np.ndarray: The blurred image.

        Behavior:
            - Ensures the blur factor is odd (required by GaussianBlur).
            - Applies Gaussian blur using cv2.GaussianBlur.
            - Returns the blurred image.
        """
        if image is None:
            return []

        if blur_factor % 2 == 0:
            blur_factor += 1
        blurred_image = cv2.GaussianBlur(
                image,
                (blur_factor, blur_factor),
                cv2.BORDER_DEFAULT
            )
        return blurred_image

    def contrast(self, image, contrast=1.4):
        """
        Apply contrast adjustment to the image.

        Args:
            image (np.ndarray): The input image to adjust contrast.
            contrast (float, optional): Contrast factor. Values >1 increase contrast, <1 decrease contrast. Defaults to 1.4.

        Returns:
            np.ndarray: The contrast-adjusted image.

        Behavior:
            - Uses cv2.convertScaleAbs to adjust contrast.
            - Returns the contrast-adjusted image.
        """
        if image is None:
            return []

        contrasted_image = cv2.convertScaleAbs(image, alpha=contrast)
        return contrasted_image

    def scaling(self, image, scale_factor=1.2):
        """
        Apply scaling augmentation to the image.

        Args:
            image (np.ndarray): The input image to be scaled.
            scale_factor (float, optional): Scaling factor. Values >1 zoom in, <1 zoom out. Defaults to 1.2.

        Returns:
            np.ndarray: The scaled image.

        Behavior:
            - Creates a scaling transformation matrix.
            - Applies the transformation using cv2.warpAffine with a white border.
            - Returns the scaled image.
        """
        if image is None:
            return []

        h, w = image.shape[:2]

        # compute the center of the image
        center_x, center_y = w / 2.0, h / 2.0

        # create the scaling transformation matrix
        # scale > 1.0 zooms in, scale < 1.0 zooms out
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale_factor)

        # choose white border value
        # (single value for grayscale, tuple for color)
        if image.ndim == 2:
            border_value = 255
        else:
            border_value = tuple([255] * image.shape[2])

        # apply the transformation, keeping original dimensions
        scaled_image = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        return scaled_image

    def illumination(self, image, brightness=60):
        """
        Apply illumination augmentation to the image.

        Args:
            image (np.ndarray): The input image to adjust illumination.
            brightness (int, optional): Brightness offset to add to pixel values. Defaults to 60.

        Returns:
            np.ndarray: The illumination-adjusted image.

        Behavior:
            - Uses cv2.convertScaleAbs to adjust brightness.
            - Returns the illumination-adjusted image.
        """
        if image is None:
            return []

        illuminated_image = cv2.convertScaleAbs(image, beta=brightness)
        return illuminated_image

    def projective(self, image, intensity=0.2):
        """
        Apply projective transformation augmentation to the image.

        Args:
            image (np.ndarray): The input image to be transformed.
            intensity (float, optional): Intensity of the perspective distortion (0.0 = none, 0.5 = strong). Defaults to 0.2.

        Returns:
            np.ndarray: The projectively transformed image.

        Behavior:
            - Defines source points (corners of the original image).
            - Applies random perspective distortion to destination points based on intensity.
            - Computes the perspective transformation matrix.
            - Applies the perspective transformation using cv2.warpPerspective with a white border.
            - Returns the projectively transformed image.
        """
        if image is None:
            return []

        h, w = image.shape[:2]

        # define source points (corners of the original image)
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        # apply random perspective distortion to destination points
        # intensity controls how much distortion (0.0 = none, 0.5 = strong)
        max_offset = int(min(w, h) * intensity)

        dst_points = np.float32([
            [np.random.randint(0, max_offset),
             np.random.randint(0, max_offset)],
            [w - 1 - np.random.randint(0, max_offset),
             np.random.randint(0, max_offset)],
            [w - 1 - np.random.randint(0, max_offset),
             h - 1 - np.random.randint(0, max_offset)],
            [np.random.randint(0, max_offset),
             h - 1 - np.random.randint(0, max_offset)]
        ])

        # compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # choose white border value
        # (single value for grayscale, tuple for color)
        if image.ndim == 2:
            border_value = 255
        else:
            border_value = tuple([255] * image.shape[2])

        # apply the perspective transformation
        projective_image = cv2.warpPerspective(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )

        return projective_image


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
        default='data/leaves',
        help='Folder to save augmented images (default: data/leaves)')
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
        '--augmentation',
        type=str,
        choices=['rotation', 'blur', 'contrast', 'scaling', 'illumination', 'projective'],
        default=None,
        help='Augmentation to apply to images (default: None)')

    return parser.parse_args()


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
            images, type_of_load = load_original_images(args.source, progress=progress, task=images_load_task)
            images = range_processing(images, range_nb=args.range_nb, range_percent=args.range_percent)
            progress.update(global_task, advance=1)

            np.random.seed(args.seed)

            # Augment images
            images_augment_task = progress.add_task("↪ Images augmentation", total=2)
            augmentator = ImgAugmentation(images)

            if type_of_load == "File":
                progress.update(images_augment_task, total=1)
            else:
                oversample_struct_task = progress.add_task("  ↪ Oversampling struct", total=0)
                augmentator.update_image_struct(progress=progress, task=oversample_struct_task)
                progress.update(images_augment_task, advance=1)

            augmenting_images_task = progress.add_task("  ↪ Augmenting images", total=0)
            augmented_images = augmentator.augment(progress=progress, task=augmenting_images_task, display=args.display, augmentation=args.augmentation)
            progress.update(images_augment_task, advance=1)
            progress.update(global_task, advance=1)

            # Save augmented images
            if args.destination not in [None, '', 'None']:
                images_save_task = progress.add_task("↪ Save augmented images", total=0)
                save_images(augmented_images, args.destination, progress=progress, task=images_save_task)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
