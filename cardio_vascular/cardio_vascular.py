import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense,  Flatten
from keras.models import load_model
class CardioVascular():
    def __init__(self):
        super().__init__()
        self.model = None
        try:
            # with open('coronary_artery.pkl', 'rb') as f:
            #     self.model = pickle.load(f)
            self.model = load_model("cardio_vascular/newcardio1.h5")
        except:
            pass

    def build_model(self, ):

        self.model = Sequential()
        self.model.add(Dense(36, activation='relu', input_shape=[11]))
        self.model.add(Dense(26, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model

    def train_model(self, X_train, y_train, X_test, y_test):
        # self.model.fit(X_train,y_train)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1)

    def evaluate_model(self, X_test, y_test):
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def prepare_input(self, sample):
        s = np.load('cardio_vascular/std.npy')
        m = np.load('cardio_vascular/mean.npy')
        new_sample = list()
        for key in sample:
            new_sample.append(sample[key])
        sample = ([new_sample] - m) / s
        return sample

    def classify(self, data):
        label = self.model.predict(data)
        return label

    def save_model(self):
        # with open('coronary_artery.pkl', 'wb') as f:
        #     pickle.dump(self.model, f)
        self.model.save("cardio_vascular_model.h5")


if __name__ == "__main__":
    classifier = CardioVascular()
    # dataset = pd.read_csv("cardio_train2.csv")
    # dataset=dataset.drop_duplicates()
    # predictors = dataset.drop("cardio", axis=1)
    # target = dataset["cardio"]
    #
    # X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.10, random_state=0)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # classifier.scaler=scaler
    # X_test = scaler.transform(X_test)
    # np.save('std.npy', scaler.var_)
    # np.save('mean.npy', scaler.mean_)

    # classifier.build_model()
    # classifier.train_model(X_train,y_train,X_test,y_test)
    # importances=classifier.model.feature_importances_
    # feature_importance_df = pd.DataFrame({'Feature': col, 'Importance': importances})
    # feature_importance_df=feature_importance_df.sort_values(by="Importance",ascending=False)
    # print(feature_importance_df)
    # classifier.save_model()
    # acc=classifier.evaluate_model(X_test,y_test)
    # print(acc)

    sample = {
        "age": 60,
        "gender": 1,
        "height": 168,
        "weight": 62,
        "ap_hi": 150,
        "ap_lo": 70,
        "cholesterol": 3,
        "gluc": 3,
        "smoke": 1,
        "alco": 1,
        "active": 1
    }
    input_data = classifier.prepare_input(sample)
    label = classifier.classify(input_data)
    print(label)


