import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

class CoronaryArtery():
    def __init__(self):
        super().__init__()
        self.model = None
        try:
            with open('coronary_artery_disease/coronary_artery.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except:
            pass

    def build_model(self, ):
        self.model = RandomForestClassifier(random_state=143)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train,y_train)


    def evaluate_model(self,X_test,y_test):
        accuracy=self.model.score(X_test,y_test)
        return accuracy

    def prepare_input(self, sample):
        sample=pd.DataFrame([sample])
        return sample

    def classify(self, data):
        label = self.model.predict_proba(data)
        return label

    def save_model(self):
        with open('coronary_artery.pkl', 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":

    classifier = CoronaryArtery()
    # dataset = pd.read_csv("heart.csv")
    # predictors = dataset.drop("target", axis=1)
    # target = dataset["target"]
    #
    # X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    # classifier.build_model()
    # classifier.train_model(X_train,y_train)
    # classifier.save_model()
    # acc=classifier.evaluate_model(X_test,y_test)
    # print(acc)

    sample = {
        "age": 60,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }
    input_data=classifier.prepare_input(sample)
    label = classifier.classify(input_data)
    print(label)


