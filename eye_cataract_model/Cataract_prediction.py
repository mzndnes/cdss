import os, glob, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import *

from tqdm import tqdm
import random
from keras.models import load_model
from keras.models import Model
import sys
import matplotlib
from keras.optimizers import Adam
import requests
from io import BytesIO
from eye_cataract_model.explain import Eye_explain as expl

epochs = 80
BATCH_SIZE = 100
IMG_HEIGHT = 192
IMG_WIDTH = 256
IMG_ROOT = 'retina/dataset/'
dense_block_size = 3
layers_in_block = 4
growth_rate = 12
classes = 2
SEED = 42


class Classify:
    def __init__(self):
        # cataract dataset
        self.IMG_DIR = [IMG_ROOT + '1_normal',
                   IMG_ROOT + '2_cataract',
                   IMG_ROOT + '2_glaucoma',
                   IMG_ROOT + '3_retina_disease']
        # ocular-disease-recognition dataset
        self.OCU_IMG_ROOT = 'ODIR-5K/ODIR-5K/Training Images/'
        self.ocu_df = 'ODIR-5K/ODIR-5K/data.xlsx'
        self.model_predict = load_model('eye_cataract_model/model/eye.hdf5', compile=False)
        self.model_test = load_model('eye_cataract_model/model/eye_epoch_50.hdf5', compile=False)

    def prepare_cataract_dataset(self):
        cat_df = pd.DataFrame(0,
                              columns=['paths',
                                       'cataract'],
                              index=range(601))

        filepaths = glob.glob(IMG_ROOT + '*/*')

        for i, filepath in enumerate(filepaths):
            filepath = os.path.split(filepath)
            cat_df.iloc[i, 0] = filepath[0] + '/' + filepath[1]

            if filepath[0] == self.IMG_DIR[0]:  # normal
                cat_df.iloc[i, 1] = 0
            elif filepath[0] == self.IMG_DIR[1]:  # cataract
                cat_df.iloc[i, 1] = 1
            elif filepath[0] == self.IMG_DIR[2]:  # glaucoma
                cat_df.iloc[i, 1] = 2
            elif filepath[0] == self.IMG_DIR[3]:  # retine_disease
                cat_df.iloc[i, 1] = 3

        # only sample normal and cataract
        cat_df = cat_df.query('0 <= cataract < 2')
        return cat_df

    def has_cataract_mentioned(self, text):
        if 'cataract' in text:
            return 1
        else:
            return 0

    def downsample(self, df):
        df = pd.concat([df.query('cataract==1'),df.query('cataract==0').sample(sum(df['cataract']),random_state=SEED)])
        return df

    def prepare_ocu_df(self):
        ocu_data = pd.read_excel(self.ocu_df)
        ocu_data['left_eye_cataract'] = ocu_data['Left-Diagnostic Keywords'] \
            .apply(lambda x: self.has_cataract_mentioned(x))
        ocu_data['right_eye_cataract'] = ocu_data['Right-Diagnostic Keywords'] \
            .apply(lambda x: self.has_cataract_mentioned(x))
        le_df = ocu_data.loc[:, ['Left-Fundus', 'left_eye_cataract']] \
            .rename(columns={'left_eye_cataract': 'cataract'})
        le_df['paths'] = self.OCU_IMG_ROOT + le_df['Left-Fundus']
        le_df = le_df.drop('Left-Fundus', axis=1)

        re_df = ocu_data.loc[:, ['Right-Fundus', 'right_eye_cataract']] \
            .rename(columns={'right_eye_cataract': 'cataract'})
        re_df['paths'] = self.OCU_IMG_ROOT + re_df['Right-Fundus']
        re_df = re_df.drop('Right-Fundus', axis=1)
        return self.downsample(le_df), self.downsample(re_df)

    def combine_df(self):
        le_df, re_df = self.prepare_ocu_df()
        ocu_df = pd.concat([le_df, re_df])
        cat_df = self.prepare_cataract_dataset()
        df = pd.concat([cat_df, ocu_df], ignore_index=True)
        return df

    def split(self, data):
        train_df, test_df = train_test_split(data,
                                             test_size=0.15,
                                             random_state=SEED,
                                             stratify=data['cataract'])
        return train_df, test_df

    def create_datasets(self, df, img_width, img_height):
        imgs = []
        for path in tqdm(df['paths']):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_width, img_height))
            imgs.append(img)

        imgs = np.array(imgs, dtype='float32')
        df = pd.get_dummies(df['cataract'])
        imgs = imgs / 255.0
        return imgs, df

#     Building the model

    def conv_layer(self, conv_x, filters):
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)
        conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
        conv_x = Dropout(0.2)(conv_x)
        return conv_x

    def dense_block(self, block_x, filters, growth_rate, layers_in_block):
        for i in range(layers_in_block):
            each_layer = self.conv_layer(block_x, growth_rate)
            block_x = concatenate([block_x, each_layer], axis=-1)
            filters += growth_rate
        return block_x, filters

    def transition_block(self, trans_x, tran_filters):
        trans_x = BatchNormalization()(trans_x)
        trans_x = Activation('relu')(trans_x)
        trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
        trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)
        return trans_x, tran_filters

    def dense_net(self, filters, growth_rate, classes, dense_block_size, layers_in_block):
        input_img = Input(shape=(192, 256, 3))
        x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)
        dense_x = BatchNormalization()(x)
        dense_x = Activation('relu')(x)
        dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
        for block in range(dense_block_size - 1):
            dense_x, filters = self.dense_block(dense_x, filters, growth_rate, layers_in_block)
            dense_x, filters = self.transition_block(dense_x, filters)
        dense_x, filters = self.dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x = BatchNormalization()(dense_x)
        dense_x = Activation('relu')(dense_x)
        dense_x = GlobalAveragePooling2D()(dense_x)
        output = Dense(classes, activation='softmax')(dense_x)
        return Model(input_img, output)

    def train_model(self):

        train_df, test_df = self.split(self.combine_df())
        train_df, val_df = self.split(train_df)
        train_imgs, train_df = self.create_datasets(train_df, IMG_WIDTH, IMG_HEIGHT)
        val_imgs, val_df = self.create_datasets(val_df, IMG_WIDTH, IMG_HEIGHT)
        # test_imgs, test_df = self.create_datasets(test_df, IMG_WIDTH, IMG_HEIGHT)
        model = self.dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_imgs, train_df, epochs=epochs, batch_size=BATCH_SIZE, shuffle=True,
                            validation_data=(val_imgs, val_df))
        model.save('eye.hdf5')
        pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
        pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
        plt.show()

    def prepare_input(self, input_image_path):
        imgs = []

        response = requests.get(input_image_path)
        if response.status_code == 200:
            # Convert the content of the response to a numpy array
            image_bytes = BytesIO(response.content)
            image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
            plot_it = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(plot_it, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, (256, 192))
            imgs.append(input_image)
            imgs = np.array(imgs, dtype='float32')
            imgs = imgs / 255.0
            return imgs, plot_it




    def predict(self,input_image_path, eye_type,enc_no, task):
        input_image, plot_it = self.prepare_input(input_image_path)
        if task == 'predict':
            prediction_float = self.model_predict.predict(input_image)
            explain_eye=expl(self.model_predict)
            file_name=explain_eye.explain(input_image,eye_type,enc_no)
            # if category == 1:
            #     label = "Normal"
            # else:
            #     label = "Cataract"
            # cv2.imshow('graycsale image',plot_it)
            # cv2.waitKey(0)
            # print(label)
            return prediction_float, file_name
        else:
            prediction = self.model_test.predict(input_image)
            category = prediction[0, 1]
            category = category.round().astype(int)
            # if category >= 0.5:
            #     label = "Not Eye"
            # else:
            #     label = "Eye"

            return category












