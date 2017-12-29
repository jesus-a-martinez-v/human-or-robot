import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle


def _load_data(path="./data/merged_train.csv"):
    return pd.read_csv(path)


def _get_features(data, drop_labels=None):
    if not drop_labels:
        return data._get_numeric_data()
    else:
        return data.drop(drop_labels, axis=1)._get_numeric_data()


def _get_features_and_labels(data):
    labels = data['outcome']
    features = _get_features(data, ['outcome'])

    return features, labels


def _oversample(features, labels):
    return SMOTE(kind='borderline1').fit_sample(features, labels)


def _scale(features):
    return preprocessing.MinMaxScaler().fit_transform(features)


def prepare_train_data():
    data = _load_data()
    features, labels = _get_features_and_labels(data)

    features, labels = _oversample(features, labels)
    features = _scale(features)
    X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.2,
                                                                    random_state=42)
    train_data = {'X_train': X_train,
                  'y_train': y_train,
                  'X_validation': X_validation,
                  'y_validation': y_validation}

    with open("train.p", "wb") as f:
        pickle.dump(train_data, f)


def prepare_test_data():
    test = _load_data("./data/merged_test.csv")
    test_features = _get_features(test)
    test_features = _scale(test_features)

    with open("test.p", "wb") as f:
        pickle.dump({'test_data': test,
                     'X_test': test_features}, f)


def prepare_train_and_test_data():
    prepare_train_data()
    prepare_test_data()


if __name__ == "__main__":
    prepare_train_and_test_data()
