import argparse as ap

from .utils.tools import load_original_images, save_images


################################################################
################################################################
#####                Transformator Class                   #####

class ImgTransformator:
    def __init__(self):
        return 

    def transform(self):
        return

    def gaussian_blur(self):
        return

    def mask(self):
        return 

    def roi_objects(self):
        return 

    def analyze_object(self):
        return

    def pseudolandmarks(self):
        return   

#####                                                      #####
################################################################






def ArgumentParsing()
    parser = ap.ArgumentParser()
    parser.add_argument('load_folder', type=str, default='../data/leaves', help='load folder')
    parser.add_argument('save_folder', type=str, default='../data/leaves_preprocessed', help='save folder')
    
    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()
        
        images = load_original_images(args.load_folder)

        transformator = ImgTransformator()
        transformed_images = transformator.transform(images)

        save_images(images, transformed_images, args.save_folder)


    except Exception as error:
        print(f'Error: {error}')