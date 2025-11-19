import cv2
import numpy as np
import argparse as ap
import plantcv as pcv



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
#                transformed_dict[class_name][image_name]['roi_objects'] = self.roi_objects(img)
#                transformed_dict[class_name][image_name]['analyze_object'] = self.analyze_object(img)
#                transformed_dict[class_name][image_name]['pseudolandmarks'] = self.pseudolandmarks(img)

        return transformed_dict


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

        # AUGMENTER LE CONTRASTE (style IRM)
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
        transformed_img = img.copy()
        return transformed_img


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
    parser.add_argument('-load_folder', type=str, default='../data/leaves', help='load folder')
    parser.add_argument('-save_folder', type=str, default='../data/leaves_preprocessed', help='save folder')

    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        images = load_original_images(args.load_folder)

        transformator = ImgTransformator()
        transformed_images = transformator.transform(images)

        save_images(transformed_images, args.save_folder)


    except Exception as error:
        print(f'Error: {error}')
