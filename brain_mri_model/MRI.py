import os
import glob
import requests
import numpy as np
import splitfolders
import nibabel as nib
from io import BytesIO
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model
# import segmentation_models_3D as sm
from keras.src.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, Dropout
from io import BytesIO
import gzip
from scipy.ndimage import zoom

EPOCHS = 50
batch_size = 5
scaler = MinMaxScaler()
kernel_initializer = 'he_uniform'


class Classify:
    def __init__(self):
        self.my_model = load_model('brain_mri_model/model/mri.hdf5', compile=False)
        self.TRAIN_DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
        self.VALIDATION_DATASET_PATH = 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
        self.input_folder = 'BraTS2020_TrainingData/input_data_3channels/'
        self.output_folder = 'BraTS2020_TrainingData/input_data_128/'

    def list_images(self):
        t2_list = sorted(glob.glob(self.TRAIN_DATASET_PATH + '*/*t2.nii'))
        t1ce_list = sorted(glob.glob(self.TRAIN_DATASET_PATH + '*/*t1ce.nii'))
        flair_list = sorted(glob.glob(self.TRAIN_DATASET_PATH + '*/*flair.nii'))
        mask_list = sorted(glob.glob(self.TRAIN_DATASET_PATH + '*/*seg.nii'))
        return t2_list, t1ce_list, flair_list, mask_list

    def prepare_array(self, t2, t1ce, flair, mask):
        for img in range(len(t2)):
            temp_image_t2 = nib.load(t2[img]).get_fdata()
            temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(
                temp_image_t2.shape)
            temp_image_t1ce = nib.load(t1ce[img]).get_fdata()
            temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(
                temp_image_t1ce.shape)
            temp_image_flair = nib.load(flair[img]).get_fdata()
            temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(
                temp_image_flair.shape)
            temp_mask = nib.load(mask[img]).get_fdata()
            temp_mask = temp_mask.astype(np.uint8)
            temp_mask[temp_mask == 4] = 3
            temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
            temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
            temp_mask = temp_mask[56:184, 56:184, 13:141]
            val, counts = np.unique(temp_mask, return_counts=True)

            if (1 - (counts[0] / counts.sum())) > 0.01:
                temp_mask = to_categorical(temp_mask, num_classes=4)
                np.save('BraTS2020_TrainingData/input_data_3channels/images/image_' + str(img) + '.npy',
                        temp_combined_images)
                np.save('BraTS2020_TrainingData/input_data_3channels/masks/mask_' + str(img) + '.npy', temp_mask)
            else:
                print("Skipped")

    def save_array(self):
        t2_list, t1ce_list, flair_list, mask_list = self.list_images()
        self.prepare_array(t2_list, t1ce_list, flair_list, mask_list)
        splitfolders.ratio(self.input_folder, output=self.output_folder, seed=42, ratio=(.75, .25), group_prefix=None)

    def load_img(self, img_dir, img_list):
        images = []
        for i, image_name in enumerate(img_list):
            if (image_name.split('.')[1]) == 'npy':
                image = np.load(img_dir + image_name)
                images.append(image)
        images = np.array(images)
        return images

    def imageLoader(self, img_dir, img_list, mask_dir, mask_list, batch_size):
        L = len(img_list)
        while True:
            batch_start = 0
            batch_end = batch_size
            while batch_start < L:
                limit = min(batch_end, L)
                X = self.load_img(img_dir, img_list[batch_start:limit])
                Y = self.load_img(mask_dir, mask_list[batch_start:limit])
                yield X, Y  # a tuple with two numpy arrays with batch_size samples
                batch_start += batch_size
                batch_end += batch_size

    def prepare_training_data(self):
        train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
        train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
        val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
        val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"
        train_img_list = os.listdir(train_img_dir)
        train_mask_list = os.listdir(train_mask_dir)
        val_img_list = os.listdir(val_img_dir)
        val_mask_list = os.listdir(val_mask_dir)
        train_img_datagen = self.imageLoader(train_img_dir, train_img_list,
                                             train_mask_dir, train_mask_list, batch_size)
        val_img_datagen = self.imageLoader(val_img_dir, val_img_list,
                                           val_mask_dir, val_mask_list, batch_size)
        return train_img_list, val_img_list, train_img_datagen, val_img_datagen, train_mask_list, val_mask_list

    def build_model(self, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
        s = inputs

        # Contraction path
        c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
        p1 = MaxPooling3D((2, 2, 2))(c1)

        c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
        p2 = MaxPooling3D((2, 2, 2))(c2)

        c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
        p3 = MaxPooling3D((2, 2, 2))(c3)

        c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
        p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

        c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

        # Expanson path
        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

        u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

        outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.summary()
        return model

    def train_model(self):
        wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
        # dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
        # focal_loss = sm.losses.CategoricalFocalLoss()
        # total_loss = dice_loss + (1 * focal_loss)
        # metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
        LR = 0.0001
        optim = keras.optimizers.Adam(LR)

        train_img_list, val_img_list, train_img_datagen, val_img_datagen, train_mask_list, val_mask_list = self.prepare_training_data()
        steps_per_epoch = len(train_img_list) // batch_size
        val_steps_per_epoch = len(val_img_list) // batch_size
        model = self.build_model(IMG_HEIGHT=128,
                                 IMG_WIDTH=128,
                                 IMG_DEPTH=128,
                                 IMG_CHANNELS=3,
                                 num_classes=4)

        # model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
        print(model.summary())

        print(model.input_shape)
        print(model.output_shape)

        history = model.fit(train_img_datagen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=val_img_datagen,
                            validation_steps=val_steps_per_epoch,
                            )

        model.save('brats_3d.hdf5')

        # plot the training and validation IoU and loss at each epoch
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def receive(self, files):
        batch = []
        for file_url in files:
            response = requests.get(file_url)
            img = response.content
            if file_url.lower().endswith('nii.gz'):
                compressed_data = BytesIO(img)
                with gzip.open(compressed_data, 'rb') as f:
                    img = f.read()
            img = nib.Nifti1Image.from_bytes(img)
            img = img.get_fdata()
            batch.append(img)
        return batch[0], batch[1], batch[2]

    def normalize(self, image):
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        return image

    def combine(self, files):
        image_1, image_2, image_3 = self.receive(files)
        combined = np.stack([self.normalize(image_1), self.normalize(image_2), self.normalize(image_3)], axis=3)
        combined = combined[56:184, 56:184, 13:141]
        input_image = combined[:, :, :, 0]
        combined = np.expand_dims(combined, axis=0)
        return combined, input_image

    def predict(self, files, e_id):
        folder_name = 'brain_mri_model/output_images/' + e_id + '/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        combined, input_image = self.combine(files)
        prediction = self.my_model.predict(combined)
        prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
        merged = np.where(prediction_argmax > 0, prediction_argmax, input_image)
        affine = np.eye(4)
        nifti_file = nib.Nifti1Image(merged, affine, dtype=np.int64)
        file_name = str(e_id) + '.nii'
        loc = folder_name + file_name
        nib.save(nifti_file, loc)
        return loc
