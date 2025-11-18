import argparse as ap

from utils.tools import load_original_images, save_images




################################################################
################################################################
#####                Augmentation Class                    #####

class ImgAugmentation:
    def __init__(self):
        return 

    def transform(self):
        return

    def rotation(self):
        return 


    def blur(self):
        return 

    def contrast(self):
        return

    def scaling(self):
        return

    def illumination(self):
        return

    def projective(self):
        return


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
        
        images = load_images(args.load_folder)

        augmentator = ImgAugmentator()
        augmented_images = augmentator.transform(images)

        save_images(images, augmented_images, args.save_folder)


    except Exception as error:
        print(f'Error: {error}')