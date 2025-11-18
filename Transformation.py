import argparse as ap

from srcs.tools import load_original_images, save_images


################################################################
################################################################
#####                Transformator Class                   #####

class ImgTransformator:
    def __init__(self):
        return

    def transform(self, img_dict):
        transformed_dict = {}

        for class_name, class_imgs in img_dict.items():
            transformed_dict[class_name] = {}
            
            for image_name, img in class_imgs.items():
                transformed_dict[class_name][image_name] = {}

                transformed_dict[class_name][image_name]['original'] = img
                transformed_dict[class_name][image_name]['gaussian_blur'] = self.gaussian_blur(img)
                transformed_dict[class_name][image_name]['mask'] = self.mask(img)
                transformed_dict[class_name][image_name]['roi_objects'] = self.roi_objects(img)
                transformed_dict[class_name][image_name]['analyze_object'] = self.analyze_object(img)
                transformed_dict[class_name][image_name]['pseudolandmarks'] = self.pseudolandmarks(img)

        return transformed_dict


    def gaussian_blur(self, img):
        tranformed_img = img.copy()
        return tranformed_img


    def mask(self, img):
        tranformed_img = img.copy()
        return tranformed_img


    def roi_objects(self, img):
        tranformed_img = img.copy()
        return tranformed_img


    def analyze_object(self, img):
        tranformed_img = img.copy()
        return tranformed_img


    def pseudolandmarks(self, img):
        tranformed_img = img.copy()
        return tranformed_img

#####                                                      #####
################################################################






def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-load_folder', type=str, default='../data/leaves', help='load folder')
    parser.add_argument('-save_folder', type=str, default='../data/leaves_preprocessed', help='save folder')
    
    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        breakpoint()
        images = load_original_images(args.load_folder)

        transformator = ImgTransformator()
        transformed_images = transformator.transform(images)

        save_images(transformed_images, args.save_folder)


    except Exception as error:
        print(f'Error: {error}')
