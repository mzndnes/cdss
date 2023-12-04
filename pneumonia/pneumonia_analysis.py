import cv2
from keras.models import Model
from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.layers import  Dense, Flatten
from keras.models import load_model
import keras
from PIL import Image


class PneumoniaAnalysis():
    def __init__(self):
        super().__init__()
        self.width = 224
        self.height = 224
        self.model = None
        try:
            self.model = load_model("pneumonia/pneumonia.keras", compile=False)
        except:
            pass

    def get_train_generator(self, image_dir,classmode="binary", shuffle=True, seed=1
                            ,batch_size=10):

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        train_generator=train_datagen.flow_from_directory(image_dir,
                                          target_size=(self.width, self.height),
                                          batch_size=batch_size,
                                          class_mode=classmode, shuffle=shuffle)
        return train_generator

    def get_test_generator(self, image_dir, classmode="binary", shuffle=False, seed=1,
                            batch_size=10):

        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        test_generator = test_datagen.flow_from_directory(image_dir,
                                                          target_size=(self.width, self.height),
                                                          batch_size=batch_size,
                                                          class_mode=classmode,shuffle=shuffle)
        return test_generator

    def build_model(self,width,height):
        base_model = VGG16(input_shape=(self.width, self.height, 3), include_top=False, weights='imagenet')

        for layer in base_model.layers:
            layer.trainable = False
        x=base_model.output
        x=Flatten()(x)
        prediction = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=base_model.input, outputs=prediction)
        # opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, train_generator,valid_generator,epochs,batch_size):
        history = self.model.fit(train_generator,
                                 validation_data=valid_generator, epochs=epochs,
                                 steps_per_epoch=len(train_generator),validation_steps=len(valid_generator))

    def evaluate_model(self, test_generator):
        # predicted_vals = self.model.predict_generator(self.test_generator, steps=len(self.test_generator))
        preds = self.model.predict(test_generator)
        return preds
        # auc_rocs = util.get_roc_curve(self.labels, predicted_vals, self.test_generator)

    def classify(self, data):
        label = self.model.predict(data)
        return label

    def prepare_input(self,img_name):
        img = keras.utils.load_img(img_name, target_size=(self.height, self.width))
        # img=Image.open(img_name).convert("RGB")

        # img = keras.utils.img_to_array(img)
        img=np.array(img)
        # img=cv2.resize(img,(self.width, self.height),interpolation = cv2.INTER_NEAREST)
        img=img/255
        img=img.reshape(1,self.width,self.height,3)
        return img

if __name__=="__main__":
    classifier=PneumoniaAnalysis()

    train_dr='./data/train'
    test_dr = './data/test'
    # train_generator=classifier.get_train_generator(train_dr)
    # test_generator=classifier.get_test_generator(test_dr)
    # print(test_generator.shape)
    # x, y = test_generator.__getitem__(0)
    # print(x[0].shape)
    # classifier.build_model()
    # classifier.train_model(train_generator,test_generator,1,16)
    # preds=classifier.evaluate_model(test_generator)
    # np.savetxt('testResult.csv', preds, delimiter=',')
    # np.savetxt('labels.csv', test_generator.labels)
    image_to_predict='person1946_bacteria_4874.jpeg'
    processed_input=classifier.prepare_input(image_to_predict)
    pred=classifier.classify(processed_input)
    print(pred)