import os
import cv2
import shutil



def load_original_images(load_folder):
    images_dict = {}

    for leaf_class in os.listdir(load_folder):
        class_folder = os.path.join(load_folder, leaf_class)
        class_basename = os.path.splitext(leaf_class)[0]

        images_dict[class_basename] = {}
        
        for image_file in os.listdir(class_folder):
            if len(image_file.split('_')) == 1:
                img_path = os.path.join(class_folder, image_file)
                images_dict[leaf_class][image_file] = cv2.imread(img_path)

    return images_dict



def load_images(load_folder): 
    images_dict = {}

    for leaf_class in os.listdir(load_folder):
        class_folder = os.path.join(load_folder, leaf_class)
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
                if images_dict[class_basename].get(image_basename) is None:
                    images_dict[leaf_class][image_basename] = {}

                image_basename, enhancement = image_basename.split('_')
                images_dict[leaf_class][image_basename][enhancement] = cv2.imread(img_path)

    return images_dict



def save_images(images, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    for class_name, class_images in images.items():
        class_folder = os.path.join(save_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        for image_name, images_variations in class_images.items():
            image_folder = os.path.join(class_folder, image_name)
            os.makedirs(image_folder, exist_ok=True)

            for variation, img in images_variations.items():
                filename = f'{image_name}_{variation}.JPG'
                filepath = os.path.join(image_folder, filename)
                
                cv2.imwrite(filepath, img)



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