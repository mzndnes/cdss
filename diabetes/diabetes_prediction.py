import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

class Classify():
    def __init__(self):
        with open("diabetes/model/diabetes_model.pkl", "rb") as file:
            self.diabetes_model = pickle.load(file)

    def prepare_data(self):
        df = pd.read_csv('Dataset of Diabetes .csv')
        df = df.drop(columns=['ID', 'No_Pation', 'Urea', 'Cr', 'TG', 'VLDL', 'HDL', 'LDL', 'Chol'])
        df['CLASS'] = df['CLASS'].replace('Y ', 'Y')
        df['CLASS'] = df['CLASS'].replace('N ', 'N')
        df['Gender'] = df['Gender'].replace('f', 'F')
        df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
        X = df.drop('CLASS', axis=1)
        y = df['CLASS']
        return X,y

    def train(self):
        X,y=self.prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier()
        model.fit(X_train_resampled, y_train_resampled)
        with open("diabetes/model/diabetes_model.pkl", "wb") as file:
            pickle.dump(model, file)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

    def predict(self, input_data):
        temp = dict()
        temp['Gender']=input_data['gender']
        temp['AGE']=input_data['age']
        temp['HbA1c']=input_data['hba1c']
        temp['BMI']=input_data['bmi']
        prediction = self.diabetes_model.predict_proba(pd.DataFrame([temp]))
        return prediction


