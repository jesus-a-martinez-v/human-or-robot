import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(path="./data/merged_train.csv"):
    return pd.read_csv(path)


def get_features_and_labels(data):
    labels = data['outcome']
    features = data.drop('outcome', axis=1)._get_numeric_data()

    return features, labels


def oversample(features, labels):
    return SMOTE(kind='borderline1').fit_sample(features, labels)


def scale(features):
    return preprocessing.MinMaxScaler().fit_transform(features)


data = load_data()
features, labels = get_features_and_labels(data)

features, labels = oversample(features, labels)
features = scale(features)

print(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(X_train.shape)
classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("Predictions: ", predictions)
print("Y_test: ", y_test)
print("Score: ", roc_auc_score(y_test, predictions, average="samples"))
print("Accuracy: ", classifier.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, predictions, pos_label=1.0))
