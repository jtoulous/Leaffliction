import os
import cv2
import pickle
import shutil

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import Sequence

from Transformation import ImgTransformator


class DetectionAgent:
    def __init__(self, transfo=['gaussian_blur'], epochs=10, train_size=0.9, img_size=(256, 256), batch_size=32):
        self.transformations = transfo
        self.epochs = epochs
        self.train_size = train_size
        self.img_size = img_size
        self.batch_size = batch_size

        self.encoder = None
        self.model = None

    def preprocessing(self, X, y=None):
        # X_resized = np.zeros((X.shape[0], self.img_size[0], self.img_size[1], X.shape[3]))
        # for i in range(X.shape[0]):
        #     X_resized[i] = cv2.resize(X[i], self.img_size)

        # # Normalisation
        # X = X_resized.astype('float32') / 255.0

        if y is not None:
            # Encodage
            self.encoder = LabelBinarizer()
            y = self.encoder.fit_transform(y)

            return X, y

        return X

    def train(self, X, y):
        X, y = self.preprocessing(X, y=y)
        train_idxs, test_idxs = self.split_indices(len(X))

        train_data_manager = DataManager(
            X=X[train_idxs],
            y=y[train_idxs],
            img_size=self.img_size,
            batch_size=self.batch_size,
            transformations=self.transformations,
            shuffle=True
        )
        test_data_manager = DataManager(
            X=X[test_idxs],
            y=y[test_idxs],
            img_size=self.img_size,
            batch_size=self.batch_size,
            transformations=[],
            shuffle=False
        )

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(len(self.encoder.classes_), activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # history = self.model.fit(
        #     train_data_manager,
        #     validation_data=test_data_manager,
        #     epochs=self.epochs,
        # ) # See if history is needed

        self.model.fit(
            train_data_manager,
            validation_data=test_data_manager,
            epochs=self.epochs,
        )

        test_loss, test_accuracy = self.model.evaluate(test_data_manager)
        print(f"Accuracy sur validation: {test_accuracy:.2%}")

    def predict(self, img):
        X = cv2.resize(img, self.img_size)
        X = X.astype('float32') / 255.0

        X = np.expand_dims(X, axis=0)

        prediction = self.model.predict(X)

        transformer = ImgTransformator()

        transformed_images = []
        for transfo in self.transformations:
            transformed_images.append(transformer.quick_use(img, transfo))

        return self.encoder.inverse_transform(prediction)[0], transformed_images

    def save(self, save_folder):
        model_file = os.path.join(save_folder, 'model.keras')
        agent_file = os.path.join(save_folder, 'agent.pkl')

        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        agent_copy_no_model = DetectionAgent(
            transfo=self.transformations,
            epochs=self.epochs,
            train_size=self.train_size,
            img_size=self.img_size,
            batch_size=self.batch_size
        )
        agent_copy_no_model.encoder = self.encoder

        with open(agent_file, 'wb') as save_file:
            pickle.dump(agent_copy_no_model, save_file)

        self.model.save(model_file)

    @staticmethod
    def load(load_folder):
        model_file = os.path.join(load_folder, 'model.keras')
        agent_file = os.path.join(load_folder, 'agent.pkl')
        agent = None

        with open(agent_file, 'rb') as a_file:
            agent = pickle.load(a_file)

        agent.model = load_model(model_file)

        return agent

    @staticmethod
    def load_from_files(model_file_path, agent_file_path):
        # Charger l'agent (sans modèle)
        with open(agent_file_path, 'rb') as f:
            agent = pickle.load(f)

        # Charger le modèle keras
        agent.model = load_model(model_file_path)

        return agent

    def split_indices(self, n_samples):
        # Séparation seulement des INDICES
        train_size = int(n_samples * self.train_size)
        indices = np.random.permutation(n_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        return train_indices, val_indices


class DataManager(Sequence):
    def __init__(self, X, y, img_size, batch_size, transformations=None, shuffle=True):
        self.X = X
        self.y = y
        self.img_size = img_size
        self.batch_size = batch_size
        self.transformations = transformations or []
        self.transformator = ImgTransformator()
        self.shuffle = shuffle

        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X_batch = []
        y_batch = []

        for i in batch_indices:
            # RESIZE et NORMALISATION ici, juste pour les images du batch
            original_img = cv2.resize(self.X[i], self.img_size)
            original_img = original_img.astype('float32') / 255.0

            X_batch.append(original_img)
            y_batch.append(self.y[i])

            # Transformées (avec resize/normal aussi)
            for transfo in self.transformations:
                transformed_img = self.transformator.quick_use(self.X[i], transfo)
                transformed_img = cv2.resize(transformed_img, self.img_size)
                transformed_img = transformed_img.astype('float32') / 255.0

                X_batch.append(transformed_img)
                y_batch.append(self.y[i])

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
