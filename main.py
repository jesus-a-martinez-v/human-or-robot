from pprint import pprint

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


candidate_classifiers = [('Decision Tree', DecisionTreeClassifier()),
                         ('Random Forest', RandomForestClassifier()),
                         ('Gradient Boosting', GradientBoostingClassifier()),
                         ('SVC', SVC()),
                         ('Gaussian NB', GaussianNB()),
                         ('Logistic Regression', LogisticRegression()),
                         ('k-Nearest Neighbors', KNeighborsClassifier()),
                         ('Perceptron', Perceptron()),
                         ('Neural Net', MLPClassifier()),
                         ('Linear Discriminant Analysis', LinearDiscriminantAnalysis())]

data = load_data()
features, labels = get_features_and_labels(data)

features, labels = oversample(features, labels)
features = scale(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(X_train.shape)

for name, classifier in candidate_classifiers:
    print(name)

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print("----------------------------------------------------------------------")
    print("Score: ", roc_auc_score(y_test, predictions))
    print("Accuracy: ", classifier.score(X_test, y_test))
    print("F1 score: ", f1_score(y_test, predictions, pos_label=1.0))
    print("----------------------------------------------------------------------\n\n")

# Grid parameter for decision tree:
dt_parameters = {
    'criterion': ('gini', 'entropy'),
    'splitter': ('best', 'random'),
    'min_samples_split': (2, 3, 5, 10),
    'min_samples_leaf': (1, 2, 5)
}

rf_parameters = {
    'n_estimators': (10, 20, 25, 50, 75, 100),
    'criterion': ('gini', 'entropy'),
    'min_samples_split': (2, 3, 5, 10),
    'min_samples_leaf': (1, 2, 5),
    'verbose': (1,),
    'n_jobs': (-1,)
}

gb_parameters = {
    'loss': ('deviance', 'exponential'),
    'learning_rate': (0.1, 0.01, 0.005),
    'n_estimators': (10, 25, 50, 75, 100),
    'criterion': ('friedman_mse', 'mse', 'mae'),
    'min_samples_split': (2, 3, 5, 10),
    'min_samples_leaf': (1, 2, 5),
    'verbose': (1,)
}

parameters_and_classifiers = [(dt_parameters, DecisionTreeClassifier(), 'Decision Tree'),
                              (rf_parameters, RandomForestClassifier(), 'Random Forest'),
                              (gb_parameters, GradientBoostingClassifier(), 'Gradient Boosting')]
scorer = make_scorer(roc_auc_score)

for parameters, classifier, name in parameters_and_classifiers:
    print("Training a", name, "classifier.")
    grid = GridSearchCV(classifier, parameters, scorer)
    grid = grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_parameters = best_estimator.get_params()

    predictions = best_estimator.predict(X_test)
    score = roc_auc_score(y_test, predictions)

    print("BEST PARAMETERS:")
    pprint(best_parameters)
    print("Tuned model has AUROC: ", score)
    print("---------------------- \n\n")
