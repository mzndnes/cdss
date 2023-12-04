import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model
import tensorflow as tf
import xray_analysis.util
class XrayAnalysis():
    def __init__(self):
        super().__init__()
        self.train_generator = None
        self.neg_weights=None
        self.pos_weights=None
        self.model=None
        try:
            self.model=load_model("xray_analysis/xrayAnalysis.h5", compile=False)
        except:
            pass
    def get_train_generator(self, train_df,image_dir, x_col, y_cols, shuffle=True, batch_size=100, seed=1, target_w=320,
                            target_h=320):

        print("getting train generator...")
        # normalize images
        image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True, dtype=np.float32)

        generator = image_generator.flow_from_dataframe(
            dataframe=train_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w, target_h))
        self.train_generator = generator
        self.neg_weights, self.pos_weights = self.compute_class_freqs(self.train_generator.labels)
        return generator

    def build_model(self):
        base_model = DenseNet121(weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(self.labels), activation="sigmoid")(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss=self.get_weighted_loss(self.pos_weights, self.neg_weights))

    def train_model(self):
        history = self.model.fit(self.train_generator,
                            validation_data=self.valid_generator,
                            steps_per_epoch=20,
                            validation_steps=25,
                            epochs=3, verbose=2)

    def evaluate_model(self,test_generator):
        # predicted_vals = self.model.predict_generator(self.test_generator, steps=len(self.test_generator))
        score,acc=self.model.predict_generator(test_generator)
        return score,acc
        # auc_rocs = util.get_roc_curve(self.labels, predicted_vals, self.test_generator)
    def get_test_generator(self,data_df, image_dir, x_col,y_cols=None,class_mode=None,
                           batch_size=16, seed=1, target_w=320, target_h=320):

        print("getting valid generators...")
        # get generator to sample dataset

        # get data sample
        batch = self.train_generator.next()
        data_sample = batch[0]

        # use sample to fit mean and std for test set generator
        image_generator = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)

        # fit generator to sample from training data
        image_generator.fit(data_sample)

        # get test generator
        data_generator = image_generator.flow_from_dataframe(
            dataframe=data_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w, target_h))


        return data_generator

    def compute_class_freqs(self,labels):
        N = labels.shape[0]
        positive_frequencies = np.sum(labels, axis=0) / N
        negative_frequencies = 1 - np.sum(labels, axis=0) / N
        return positive_frequencies, negative_frequencies

    def get_weighted_loss(self,pos_weights, neg_weights, epsilon=1e-7):

        def weighted_loss(y_true, y_pred):

            # initialize loss to zero
            y_true = tf.cast(y_true, tf.float32)

            loss = 0.0

            for i in range(len(pos_weights)):
                # for each class, add average weighted loss for that class
                loss += K.mean(-(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)
                                 + neg_weights[i] * (1 - y_true[:, i]) * K.log(
                            1 - y_pred[:, i] + epsilon)))  # complete this line
            return loss

        return weighted_loss

    def classify(self,data):
        label = self.model.predict(data)
        return label


if __name__=="__main__":

    c=XrayAnalysis()
    train_df = pd.read_csv("nih/train-all.csv")

    labels = ['Cardiomegaly','Emphysema','Effusion','Hernia',
                      'Infiltration','Mass','Nodule','Atelectasis',
                      'Pneumothorax','Pleural_Thickening','Pneumonia',
                      'Fibrosis','Edema','Consolidation']

    valid_df = pd.read_csv("nih/valid-all.csv")
    test_df = pd.read_csv("nih/test.csv")
    IMAGE_DIR = "nih/images/"
    c.get_train_generator(train_df,IMAGE_DIR, "Image", labels)

    test_generator=c.get_test_generator( test_df,IMAGE_DIR, "Image")

    score,acc=c.evaluate_model(test_generator)
    print(score,acc)
    # x = test_generator.__getitem__(0)
    # img = x[0].reshape(1, 320, 320, 3)
    # pred = c.classify(img)

    # xvalues = np.arange(len(pred[0]))
    #
    # fig = plt.figure(figsize=(10, 5))
    #
    #
    # plt.bar(xvalues, pred[0], color='maroon', width=0.4)
    # plt.xticks(xvalues, labels, rotation=90)
    # plt.xlabel("Probable Diseases")
    # plt.ylabel("Probability")
    # plt.title("Disease diagonosis")
    # plt.legend()
    # plt.show()


