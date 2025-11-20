import os
import cv2


def load_original_images(load_folder, progress=None, task=None):
    """
    Load original images from the specified folder structure.

    Args:
        load_folder (str): Path to the folder or image file.
        progress (Progress, optional): Progress tracking object.
        task (Task, optional): Task identifier for progress tracking.

    Returns:
        dict: A nested dictionary with structure {class_name: {image_file: image_data}}
    """
    images_dict = {}

    # Case 1: Direct image file path (leaves/category/imagefile)
    if os.path.isfile(load_folder):
        parent_dir = os.path.dirname(load_folder)
        class_name = os.path.basename(parent_dir)
        image_file = os.path.basename(load_folder)

        if task is not None:
            progress.update(task, total=1)

        images_dict[class_name] = {}
        images_dict[class_name][image_file] = cv2.imread(load_folder)

        if task is not None:
            progress.update(task, advance=1)

        return images_dict, "File"

    # Case 2: Category folder path (leaves/category)
    if os.path.isdir(load_folder):
        contents = os.listdir(load_folder)
        if contents and all(os.path.isfile(os.path.join(load_folder, item)) for item in contents):
            class_name = os.path.basename(load_folder.rstrip('/\\'))

            if task is not None:
                progress.update(task, total=len([f for f in contents if len(f.split('_')) == 1]))

            images_dict[class_name] = {}

            for image_file in contents:
                if len(image_file.split('_')) == 1:
                    img_path = os.path.join(load_folder, image_file)
                    images_dict[class_name][image_file] = cv2.imread(img_path)

                    if task is not None:
                        progress.update(task, advance=1)

            return images_dict, "Category"

    # Case 3: Root folder path (leaves/)
    if task is not None:
        progress.update(task, total=sum(len([f for f in os.listdir(os.path.join(load_folder, leaf_class)) if len(f.split('_')) == 1]) for leaf_class in os.listdir(load_folder) if os.path.isdir(os.path.join(load_folder, leaf_class))))

    for leaf_class in os.listdir(load_folder):
        class_folder = os.path.join(load_folder, leaf_class)

        if not os.path.isdir(class_folder):
            continue

        class_basename = os.path.splitext(leaf_class)[0]

        images_dict[class_basename] = {}

        for image_file in os.listdir(class_folder):
            if len(image_file.split('_')) == 1:
                img_path = os.path.join(class_folder, image_file)
                images_dict[leaf_class][image_file] = cv2.imread(img_path)

                if task is not None:
                    progress.update(task, advance=1)

    return images_dict, "Root"


def load_images(load_folder, progress=None, task=None):
    """
    Load all images from the specified folder structure.

    Args:
        load_folder (str): Path to the folder or image file.
        progress (Progress, optional): Progress tracking object.
        task (Task, optional): Task identifier for progress tracking.

    Returns:
        dict: A nested dictionary with structure {class_name: {image_file: {enhancement_type: image_data}}}
    """
    images_dict = {}

    # Case 1: Direct image file path (leaves/category/imagefile)
    if os.path.isfile(load_folder):
        parent_dir = os.path.dirname(load_folder)
        class_name = os.path.basename(parent_dir)
        image_file = os.path.basename(load_folder)
        image_basename = os.path.splitext(image_file)[0]

        if task is not None:
            progress.update(task, total=1)

        images_dict[class_name] = {}

        if len(image_basename.split('_')) == 1:
            images_dict[class_name][image_basename] = {}
            images_dict[class_name][image_basename]['original'] = cv2.imread(load_folder)
        else:
            base_name, enhancement = image_basename.split('_', 1)
            if images_dict[class_name].get(base_name) is None:
                images_dict[class_name][base_name] = {}
            images_dict[class_name][base_name][enhancement] = cv2.imread(load_folder)

        if task is not None:
            progress.update(task, advance=1)

        return images_dict, "File"

    # Case 2: Category folder path (leaves/category)
    if os.path.isdir(load_folder):
        contents = os.listdir(load_folder)
        if contents and all(os.path.isfile(os.path.join(load_folder, item)) for item in contents):
            class_name = os.path.basename(load_folder.rstrip('/\\'))

            if task is not None:
                progress.update(task, total=len(contents))

            images_dict[class_name] = {}

            for image_file in contents:
                img_path = os.path.join(load_folder, image_file)
                image_basename = os.path.splitext(image_file)[0]

                if len(image_basename.split('_')) == 1:
                    if images_dict[class_name].get(image_basename) is None:
                        images_dict[class_name][image_basename] = {}
                    images_dict[class_name][image_basename]['original'] = cv2.imread(img_path)
                else:
                    base_name, enhancement = image_basename.split('_', 1)
                    if images_dict[class_name].get(base_name) is None:
                        images_dict[class_name][base_name] = {}
                    images_dict[class_name][base_name][enhancement] = cv2.imread(img_path)

                if task is not None:
                    progress.update(task, advance=1)

            return images_dict, "Category"

    # Case 3: Root folder path (leaves/)
    if task is not None:
        progress.update(task, total=sum(len(os.listdir(os.path.join(load_folder, leaf_class))) for leaf_class in os.listdir(load_folder) if os.path.isdir(os.path.join(load_folder, leaf_class))))

    for leaf_class in os.listdir(load_folder):
        class_folder = os.path.join(load_folder, leaf_class)

        if not os.path.isdir(class_folder):
            continue

        class_basename = os.path.splitext(leaf_class)[0]

        images_dict[class_basename] = {}

        for image_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, image_file)
            image_basename = os.path.splitext(image_file)[0]

            if len(image_basename.split('_')) == 1:

                if images_dict[class_basename].get(image_basename) is None:
                    images_dict[leaf_class][image_basename] = {}

                images_dict[leaf_class][image_basename]['original'] = cv2.imread(img_path)

            else:
                base_name, enhancement = image_basename.split('_', 1)
                if images_dict[class_basename].get(base_name) is None:
                    images_dict[leaf_class][base_name] = {}
                images_dict[leaf_class][base_name][enhancement] = cv2.imread(img_path)

            if task is not None:
                progress.update(task, advance=1)

    return images_dict, "Root"


def save_images(images, save_folder, progress=None, task=None):
    """
    Save images to the specified folder structure.

    Args:
        images (dict): A nested dictionary with structure {class_name: {image_file: {enhancement_type: image_data}}}
        save_folder (str): Path to the folder where images will be saved.
        progress (Progress, optional): Progress tracking object.
        task (Task, optional): Task identifier for progress tracking.
    """
    os.makedirs(save_folder, exist_ok=True)

    if task is not None:
        progress.update(task, total=sum(len(variations) for class_images in images.values() for variations in class_images.values()))

    for class_name, class_images in images.items():
        class_folder = os.path.join(save_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for image_name, images_variations in class_images.items():
            for variation, img in images_variations.items():
                if variation == 'original':
                    filename = f'{image_name.rstrip(".JPG")}.JPG'
                else:
                    filename = f'{image_name.rstrip(".JPG")}_{variation}.JPG'
                filepath = os.path.join(class_folder, filename)

                cv2.imwrite(filepath, img)

                if task is not None:
                    progress.update(task, advance=1)


# loaded original imgs
#{
#    'Apple_Black_rot': {
#        'image (1)': img_1,
#        'image (2)': img_2,
#        'image (3)': img_3,
#    }
#
#    'Apple_healthy': {
#        'image (1)': img_1,
#        'image (2)': img_2,
#        'image (3)': img_3,
#    }
#
#    'etc...': {
#        '...': ...
#    }
#
#}
#
#
#
## loaded imgs
#{
#    'Apple_Black_rot': {
#        'image (1)': {
#            'original': img_1_original,
#            'rotation': img_1_rotation,
#            'blur': img_1_blur,
#            'etc...': ...
#        },
#
#        'image (2)': {
#            'original': img_2_original,
#            'rotation': img_2_rotation,
#            'blur': img_2_blur,
#            'etc...': ...
#        },
#
#        'etc...':{
#            'etc...': ...
#        }
#    }
#
#    'Apple_healthy': {
#        'image (1)': {
#            'original': img_1_original,
#            'rotation': img_1_rotation,
#            'blur': img_1_blur,
#            'etc...': ...
#        },
#
#        'image (2)': {
#            'original': img_2_original,
#            'rotation': img_2_rotation,
#            'blur': img_2_blur,
#            'etc...': ...
#        },
#
#        'etc...':{
#            'etc...': ...
#        }
#    }
#
#
#    'etc...': {
#        '...': ...
#    }
#
#}
#
#
## saved images
#{
#    'Apple_Black_rot': {
#        'image (1)': {
#            'original': img_1_original,
#            'rotation': img_1_rotation,
#            'blur': img_1_blur,
#            'etc...': ...
#        },
#
#        'image (2)': {
#            'original': img_2_original,
#            'rotation': img_2_rotation,
#            'blur': img_2_blur,
#            'etc...': ...
#        },
#
#        'etc...':{
#            'etc...': ...
#        }
#    }
#
#    'Apple_healthy': {
#        'image (1)': {
#            'original': img_1_original,
#            'rotation': img_1_rotation,
#            'blur': img_1_blur,
#            'etc...': ...
#        },
#
#        'image (2)': {
#            'original': img_2_original,
#            'rotation': img_2_rotation,
#            'blur': img_2_blur,
#            'etc...': ...
#        },
#
#        'etc...':{
#            'etc...': ...
#        }
#    }
#
#
#    'etc...': {
#        '...': ...
#    }
#}
