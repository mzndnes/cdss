
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
class RetinoPathy():
    def __init__(self):
        super().__init__()
        self.std_dev=None
        self.mean=None
        self.model= pickle.load(open('retinopathy/retinopathy.pkl', 'rb'))
        try:
            with open('retinopathy/retinopathy.pkl', 'rb') as f:
                self.model = pickle.load(f)
            # self.std_mean=pd.from_csv('trainmean1.csv')
            # self.std_dev = pd.from_csv('trainstddev1.csv')
            # self.mean=self.mean.transpose()[0]
            # self.std_dev = self.std_dev.transpose()[0]

        except:
            pass
    def standardize_train_test(self,df_train, df_test):

        df_train_log = np.log(df_train)
        df_test_log = np.log(df_test)

        self.mean = df_train_log.mean(axis=0)
        self.std_dev = df_train_log.std(axis=0)
        df_train_standard = (df_train_log - self.mean) / self.std_dev
        df_test_standard = (df_test_log - self.mean) / self.std_dev

        return df_train_standard, df_test_standard

    def standardize_data(self,df):
        df_log = np.log(df)
        df_standard = (df_log - self.mean) / self.std_dev
        return df_standard

    def build_model(self,):
        self.model = LogisticRegression()

    def train_model(self,X_train,y_train):
        history = self.model.fit(X_train,y_train)
        print(history)

    def evaluate_model(self,X_test,y_test):
        y_pred = self.model.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        return acc_score

    def classify(self,data):
        label=self.model.predict_proba(data)
        # print(label)
        # print(self.model.classes_)
        return label

    def add_features(self,df):

        features = df.columns
        m = len(features)
        new_df = df.copy(deep=True)

        for i in range(m):

            feature_i_name = features[i]

            feature_i_data = df.loc[:, feature_i_name]

            # choose the index of column 'j' to be greater than column i
            for j in range(i + 1, m):
                # get the name of feature 'j'
                feature_j_name = features[j]

                # get the data for feature j'
                feature_j_data = df.loc[:, feature_j_name]

                feature_i_j_name = f"{feature_i_name}_x_{feature_j_name}"

                new_df[feature_i_j_name] = feature_i_data * feature_j_data


        return new_df
if __name__=="__main__":
    classifier = RetinoPathy()
    X = pd.read_csv('X_data.csv', index_col=0)
    y_df = pd.read_csv('y_data.csv', index_col=0)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y_df, train_size=0.75, random_state=0)
    X_train,X_test=classifier.standardize_train_test(X_train_raw,X_test_raw)

    # X_train=classifier.add_features(X_train)
    # X_test = classifier.add_features(X_test)
    # classifier.build_model()
    # classifier.train_model(X_train,y_train)
    # acc_score=classifier.evaluate_model(X_test,y_test)

    label=classifier.classify(X_test.iloc[0:1,:])
    print(label)



