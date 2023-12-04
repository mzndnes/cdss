import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,  Flatten
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import Sequential
import pandas as pd
from keras.utils import load_img, img_to_array



class ChestXrayRecognition():
    def __init__(self):
        super().__init__()
        self.width = 224
        self.height = 224
        self.model = load_model("chest_xray_recognition/chest_xray_recognition_model.h5", compile=False)
        try:
            self.model = load_model("chest_xray_recognition/chest_xray_recognition_model.keras", compile=False)
        except:
            pass

    def get_train_generator(self, train_df, image_dir, x_col, y_cols=None, classmode=None, shuffle=True, batch_size=32,
                            seed=1):

        print("getting train generator...")
        # normalize images
        image_generator = ImageDataGenerator(rescale=1. / 255, dtype=np.float32)

        generator = image_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode=classmode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(self.width, self.height))

        return generator

    def get_test_generator(self, train_df, image_dir, x_col, y_cols=None, classmode=None, shuffle=False, batch_size=8,
                           seed=1):

        print("getting test generator...")
        # normalize images
        image_generator = ImageDataGenerator(rescale=1. / 255, dtype=np.float32)

        generator = image_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode=classmode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(self.width, self.height))

        return generator

    def build_model(self):
        base_model = VGG16(input_shape=(self.width, self.height, 3), include_top=False, weights='imagenet')

        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Flatten()(x)
        prediction = Dense(6, activation="softmax")(x)
        self.model = Model(inputs=base_model.input, outputs=prediction)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(self.model.summary())

    def train_model(self, train_generator, test_generator, batch_size, epochs):
        history = self.model.fit(train_generator, validation_data=test_generator, epochs=epochs, batch_size=batch_size)
        self.model.save('chect_xray_recognition.h5')

    def evaluate_model(self, test_generator):
        preds = self.model.predict(test_generator)
        return preds

    def classify(self, data):
        label = self.model.predict(data)
        return label

    def prepare_input(self, image):
        # PIL.Image.frombytes(mode, size, data, decoder_name='raw', *args)
        my_image = load_img(image, target_size=(self.width, self.height))
        my_image = img_to_array(my_image)
        my_image = my_image / 255
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        return my_image


if __name__ == "__main__":
    classifier = ChestXrayRecognition()
    images_dr = 'data/images/'

    train_df = pd.read_csv("xray_recogntion_train.csv", dtype=str)
    test_df = pd.read_csv("xray_recogntion_test.csv", dtype=str)
    train_generator = classifier.get_train_generator(train_df, images_dr, 'Image', 'label', "categorical")
    test_generator = classifier.get_test_generator(test_df, images_dr, 'Image', 'label', "categorical")
    # x, y = test_generator.__getitem__(0)
    # print(x[0].shape)
    # plt.imshow(x[0])
    # classifier.build_model()
    # classifier.train_model(train_generator, test_generator, 1, 1)
    # classifier.model.save('newxray0.keras')
    # for i in range(10):
    #     classifier.model=load_model('newxray'+str(i)+'.keras')
    #     classifier.train_model(train_generator, test_generator, 1, 1)
    #     classifier.model.save('newxray'+str(i+1)+'.keras')
    # preds = classifier.evaluate_model(test_generator)
    # preds=np.argmax(preds,axis=1)
    # preds=preds.T
    # np.savetxt('testResult.csv', preds, delimiter=',')
    # print(preds)

    image='person1946_bacteria_4874.jpeg'
    processed_img=classifier.process_image(image)
    pred=classifier.classify(processed_img)
    print(pred)