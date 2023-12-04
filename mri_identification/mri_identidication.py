import os
import nibabel as nib
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from scipy.ndimage import zoom
import requests
from io import BytesIO
import gzip

class1_path = 'MRI_identification/take'
class2_path = 'MRI_identification/reject'
batch_size = 4

class identify():
    def __init__(self):
        self.model= load_model('mri_identification/models/mri_identification_small.h5', compile=False)

    def data_generator(self,folder_path, labels, batch_size):
        num_samples = len(labels)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        while True:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                batch_data = []
                for idx in batch_indices:
                    filename = os.listdir(folder_path)[idx]
                    file_path = os.path.join(folder_path, filename)
                    img = nib.load(file_path).get_fdata()
                    batch_data.append(img)
                batch_data = normalize(np.array(batch_data))  # Normalization
                batch_labels = labels[batch_indices]

                yield batch_data, batch_labels

    def make_labels(self, path1,path2):
        class1_labels = np.zeros(len(os.listdir(path1)))
        class2_labels = np.ones(len(os.listdir(path2)))
        train_generator_class1 = self.data_generator(path1, class1_labels, batch_size)
        train_generator_class2 = self.data_generator(path2, class2_labels, batch_size)
        return train_generator_class1, train_generator_class2

    def merged_generators(self,gen1, gen2):
        while True:
            data1, labels1 = next(gen1)
            data2, labels2 = next(gen2)
            yield np.concatenate((data1, data2), axis=0), np.concatenate((labels1, labels2), axis=0)

    def create_generator(self):
        train_generator_class1, train_generator_class2 = self.make_labels( class1_path,class2_path)
        train_generator = self.merged_generators(train_generator_class1, train_generator_class2)
        return train_generator


    def create_model(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(128, 128, 128, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def train_model(self):
        model=self.create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        epochs = 10
        steps_per_epoch = min(len(os.listdir(class1_path)), len(os.listdir(class2_path))) // batch_size
        model.fit(self.create_generator(), steps_per_epoch=steps_per_epoch, epochs=epochs)

    def preprocess(self,files, target_shape):
        batch=[]
        for file_url in files:
            response=requests.get(file_url)
            img=response.content
            if file_url.lower().endswith('nii.gz'):
                compressed_data = BytesIO(img)
                with gzip.open(compressed_data, 'rb') as f:
                    img = f.read()
            img = nib.Nifti1Image.from_bytes(img)
            img = img.get_fdata()
            current_shape = img.shape
            if current_shape != target_shape:
                zoom_factors = (
                    target_shape[0] / current_shape[0],
                    target_shape[1] / current_shape[1],
                    target_shape[2] / current_shape[2]
                )
                img = zoom(img, zoom_factors, order=1, mode='nearest')
            batch.append(img)
        batch = normalize(np.array(batch))  # Normalize the data
        return batch

    def predict(self,images):
        batch = self.preprocess(images,(128,128,128))
        prediction=self.model.predict(batch)
        class_predictions = [0 if pred > 0.5 else 1 for pred in prediction]
        return class_predictions

