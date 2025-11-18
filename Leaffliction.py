from srcs.ImgClassificator import ImgClassificator
from srcs.ImgTransformator import ImgTransformator




def Parsing():
    parser = ap.ArgumentParser()
    parser.add_argument()

    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = Parsing()




    except Exception as error:
        print(f'Error: {error}')