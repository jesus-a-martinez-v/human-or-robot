import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


def run_on_test_set():
    with open("best.p", "rb") as f:
        best_classifier = pickle.load(f)

    with open("test.p", "rb") as f:
        data = pickle.load(f)
        X_test = data['X_test']
        test_data = data['test_data']

    test_data['prediction'] = best_classifier.predict(X_test)

    print("Submission created")
    test_data[['bidder_id', 'prediction']].to_csv('./data/submission.csv', index=False)


def run_logistic_regression_on_test_set():
    clf = LogisticRegression()

    with open("test.p", "rb") as f:
        data = pickle.load(f)
        X_test = data['X_test']
        test_data = data['test_data']

    with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']

    clf.fit(X_train, y_train)

    test_data['prediction'] = clf.predict(X_test)

    print("Submission created")
    test_data[['bidder_id', 'prediction']].to_csv('./data/lr_submission.csv', index=False)


def run_predict_majority_class_on_test_set():
    with open("test.p", "rb") as f:
        data = pickle.load(f)
        X_test = data['X_test']
        test_data = data['test_data']

    test_data['prediction'] = np.zeros(X_test.shape[0])

    print("Submission created")
    test_data[['bidder_id', 'prediction']].to_csv('./data/majority_submission.csv', index=False)


if __name__ == "__main__":
    run_on_test_set()
    run_logistic_regression_on_test_set()
    run_predict_majority_class_on_test_set()
