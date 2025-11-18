import os
import cv2




def load_original_images(load_folder):
    images_dict = {}

    for leaf_class in os.listdir(load_folder):
        images_dict[leaf_class] = {}
        class_folder = os.path.join()
        
        for image_file in os.listdir(os.path.join(load_folder, leaf_class)):
            if len(image_file.split('_')) == 1:
                img_path = os.path.join()
                images_dict[leaf_class][image_file] = 

    return images_dict



def load_images(imgs): 
    return



def save_images(imgs, save_folder):
    return





# loaded original imgs
{
    'Apple_Black_rot': {
        'image (1)': img_1,
        'image (2)': img_2,
        'image (3)': img_3,
    }

    'Apple_healthy': {
        'image (1)': img_1,
        'image (2)': img_2,
        'image (3)': img_3,
    }

    'etc...': {
        '...': ...
    }

}



# loaded imgs
{
    'Apple_Black_rot': {
        'image (1)': {
            'original': img_1_original,
            'rotation': img_1_rotation,
            'blur': img_1_blur,
            'etc...': ...
        },
        
        'image (2)': {
            'original': img_2_original,
            'rotation': img_2_rotation,
            'blur': img_2_blur,
            'etc...': ...
        },
        
        'etc...':{
            'etc...': ...
        }
    }

    'Apple_healthy': {
        'image (1)': {
            'original': img_1_original,
            'rotation': img_1_rotation,
            'blur': img_1_blur,
            'etc...': ...
        },
        
        'image (2)': {
            'original': img_2_original,
            'rotation': img_2_rotation,
            'blur': img_2_blur,
            'etc...': ...
        },
        
        'etc...':{
            'etc...': ...
        }
    }


    'etc...': {
        '...': ...
    }

}


# saved images
{
    'Apple_Black_rot': {
        'image (1)': {
            'original': img_1_original,
            'rotation': img_1_rotation,
            'blur': img_1_blur,
            'etc...': ...
        },
        
        'image (2)': {
            'original': img_2_original,
            'rotation': img_2_rotation,
            'blur': img_2_blur,
            'etc...': ...
        },
        
        'etc...':{
            'etc...': ...
        }
    }

    'Apple_healthy': {
        'image (1)': {
            'original': img_1_original,
            'rotation': img_1_rotation,
            'blur': img_1_blur,
            'etc...': ...
        },
        
        'image (2)': {
            'original': img_2_original,
            'rotation': img_2_rotation,
            'blur': img_2_blur,
            'etc...': ...
        },
        
        'etc...':{
            'etc...': ...
        }
    }


    'etc...': {
        '...': ...
    }
}