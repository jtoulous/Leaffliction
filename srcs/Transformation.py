import argparse as ap




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
    parser.add_argument('load_folder', type=str, help='load folder')
    parser.add_argument('save_folder', type=str, help='save folder')
    
    return parser.parse_args()




def load_images(load_folder):
    return



def save_images(save_folder):
    return




if __name__ == '__main__':
    try:
        args = ArgumentParsing()
        
        images = load_images(args.load_folder)

        transformator = ImgTransformator()
        transformed_images = transformator.transform(images)

        save_images(images, transformed_images, args.save_folder)


    except Exception as error:
        print(f'Error: {error}')