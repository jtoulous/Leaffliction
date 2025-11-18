import cv2
import numpy as np
import argparse as ap

from srcs.tools import load_original_images, save_images
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn # type: ignore


class ImgAugmentation:
    def __init__(self, images_structure):
        self.images_structure = images_structure

        return

    def transform(self, image=None, progress=None, task=None):
        if image:
            self.rotation(image)
            self.blur(image)
            self.contrast(image)
            self.scaling(image)
            self.illumination(image)
            self.projective(image)
        else:
            if task is not None:
                progress.update(task, total=len(self.images_structure))

            for category in self.images_structure:

                if task is not None:
                    progress.update(task, description=f"↪ Augmenting images: {category}")

                for img_key in self.images_structure[category]:
                    image = self.images_structure[category][img_key]
                    self.images_structure[category][img_key] = {
                        'original': image,
                        'rotation': self.rotation(image),
                        'blur': self.blur(image),
                        'contrast': self.contrast(image),
                        'scaling': self.scaling(image),
                        'illumination': self.illumination(image),
                        'projective': self.projective(image),
                    }

                if task is not None:
                    progress.update(task, advance=1)
                    progress.update(task, description=f"↪ Images augmentation")

        return self.images_structure

    def rotation(self, image, angle=np.random.randint(20, 340)):
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

    def blur(self, image, blur_factor=15):
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
        if image is None:
            return []

        contrasted_image = cv2.convertScaleAbs(image, alpha=contrast)
        return contrasted_image

    def scaling(self, image, scale_factor=1.2):
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
        if image is None:
            return []

        illuminated_image = cv2.convertScaleAbs(image, beta=brightness)
        return illuminated_image

    def projective(self, image, intensity=0.2):
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
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--load_folder',
        type=str,
        default='data/leaves',
        help='Folder with original images (default: data/leaves)')
    parser.add_argument(
        '--save_folder',
        type=str,
        default='data/leaves_preprocessed',
        help='Folder to save augmented images \
              (default: data/leaves_preprocessed)')
    parser.add_argument(
        '--range',
        type=int,
        default=100,
        help='Percentage of augmented images to process (default: 100)')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)')

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
            images = load_original_images(args.load_folder, progress=progress, task=images_load_task)
            images = {cat: dict(list(imgs.items())[:int(len(imgs)
                    * args.range / 100)]) for cat, imgs in images.items()}
            progress.update(global_task, advance=1)

            np.random.seed(args.seed)

            # Augment images
            images_augment_task = progress.add_task("↪ Images augmentation", total=0)
            augmentator = ImgAugmentation(images)
            augmented_images = augmentator.transform(progress=progress, task=images_augment_task)
            progress.update(global_task, advance=1)

            # Save augmented images
            images_save_task = progress.add_task("↪ Save augmented images", total=0)
            save_images(augmented_images, args.save_folder, progress=progress, task=images_save_task)
            progress.update(global_task, advance=1)

    except Exception as error:
        print(f'Error: {error}')
