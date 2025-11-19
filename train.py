import cv2
import numpy as np
import argparse as ap
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from Augmentation import ImgAugmentation
from Transformation import ImgTransformator

from srcs.tools import load_original_images


import datetime


class DetectionAgent:
    def __init__(self, test_size=0.2, img_size=(256, 256)):
        self.test_size = test_size
        self.img_size = img_size

        self.encoder = None
        self.model = None


    def preprocessing(self, mode, original_images):
        if mode == 'train':
            augmentator = ImgAugmentation(deepcopy(original_images))
            transformator = ImgTransformator(deepcopy(original_images))

            augmented_imgs = augmentator.augment()
            transformed_imgs = transformator.transform()

            X_list = []
            y_list = []

            for classe in augmented_imgs:
                for img_name in augmented_imgs[classe]:
                    for transform_name, img in augmented_imgs[classe][img_name].items():
                        resized_img = cv2.resize(img, self.img_size)
                        X_list.append(resized_img)
                        y_list.append(classe)

            X = np.array(X_list)
            y = np.array(y_list)

            ## Normalisation
            X = X.astype('float32') / 255.0

            ## Encodage
            self.encoder = LabelBinarizer()
            y = self.encoder.fit_transform(y)

            return train_test_split(X, y, test_size=self.test_size, stratify=y)


        elif mode == 'predict':
            resized_img = cv2.resize(original_images, self.img_size)            

            X = np.array([resized_img])
            
            ## Normalisation
            X = X.astype('float32') / 255.0

            return X

            
    def train(self, original_images):
        breakpoint()
        
        start_time = datetime.datetime.now()
        print(f"🚀 Début preprocessing: {start_time.strftime('%H:%M')}")
        
        X_train, X_test, y_train, y_test = self.preprocessing('train', original_images)
        
        end_time = datetime.datetime.now()
        print(f"✅ Fin preprocessing: {end_time.strftime('%H:%M')}")

        duration = end_time - start_time
        print(f"⏱️  Durée: {duration}")

        breakpoint()
    
        # 1. CRÉATION DU MODÈLE CNN  
        self.model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'), 
            MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(len(self.encoder.classes_), activation='softmax')
        ])

        # 2. COMPILATION
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

        # 3. ENTRAÎNEMENT
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32
        )

        # 4. ÉVALUATION
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Accuracy sur validation: {test_accuracy:.2%}")



    def predict(self, original_img):
        X = self.preprocessing('predict', original_img)

    def save(self, save_folder):
        pass





def ArgumentParsing():
    parser = ap.ArgumentParser()
    parser.add_argument('-imgs_folder', default='data/leaves', help='original images folder')
    parser.add_argument('-save_folder', default='Leaffliction_results', help='output file')
    return parser.parse_args()





if __name__ == '__main__':
    try:
        args = ArgumentParsing()

        original_images = load_original_images(args.imgs_folder)
        agent = DetectionAgent()
        agent.train(original_images)
        agent.save(args.save_folder)


    except Exception as error:
        print(f'Error: {error}')