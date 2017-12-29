import pickle
from operator import itemgetter
from pprint import pprint

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_candidates():
    candidate_classifiers = [{'name': 'Decision Tree',
                              'classifier': DecisionTreeClassifier(),
                              'grid': {'criterion': ('gini', 'entropy'),
                                       'splitter': ('best', 'random'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5)},
                              'score': 0.0},
                             {'name': 'Random Forest',
                              'classifier': RandomForestClassifier(),
                              'grid': {'n_estimators': (10, 20, 25, 50, 75, 100),
                                       'criterion': ('gini', 'entropy'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5),
                                       'verbose': (1,),
                                       'n_jobs': (-1,)},
                              'score': 0.0},
                             {'name': 'Gradient Boosting',
                              'classifier': GradientBoostingClassifier(),
                              'grid': {'loss': ('deviance', 'exponential'),
                                       'learning_rate': (0.1, 0.01, 0.005),
                                       'n_estimators': (10, 25, 50, 75, 100),
                                       'criterion': ('friedman_mse', 'mse', 'mae'),
                                       'min_samples_split': (2, 3, 5, 10),
                                       'min_samples_leaf': (1, 2, 5),
                                       'verbose': (1,)},
                              'score': 0.0},
                             {'name': 'SVC',
                              'classifier': SVC(),
                              'grid': {'kernel': ('rbf', 'poly', 'linear', 'sigmoid'),
                                       'degree': (3, 4, 5),
                                       'C': (1.0, 0.5, 0.001, 1.5)},
                              'score': 0.0},
                             {'name': 'Gaussian NB',
                              'classifier': GaussianNB(),
                              'grid': {},
                              'score': 0.0},
                             {'name': 'Logistic Regression',
                              'classifier': LogisticRegression(),
                              'grid': {'C': (1.0, 0.5, 0.001, 1.5)},
                              'score': 0.0},
                             {'name': 'k-Nearest Neighbors',
                              'classifier': KNeighborsClassifier(),
                              'grid': {'n_neighbors': (2, 3, 5, 10, 50),
                                       'weights': ('uniform', 'distance'),
                                       'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                                       'p': (1, 2, 3)},
                              'score': 0.0},
                             {'name': 'Perceptron',
                              'classifier': Perceptron(),
                              'grid': {'tol': (1e-4, 1e-3),
                                       'max_iter': (1000, 5000),
                                       'penalty': (None, 'l1', 'l2')},
                              'score': 0.0},
                             {'name': 'Neural Net',
                              'classifier': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(512, 128)),
                              'grid': {'solver': ('lbfgs', 'sgd', 'adam'),
                                       'hidden_layer_sizes': [(128, 64), (512, 128, 64)],
                                       'activation':  ('identity', 'logistic', 'tanh', 'relu'),
                                       'alpha': (0.002, 0.005, 0.0001),
                                       'learning_rate': ('constant', 'invscaling', 'adaptive')},
                              'score': 0.0},
                             {'name': 'Linear Discriminant Analysis',
                              'classifier': LinearDiscriminantAnalysis(),
                              'grid': {'solver': ('svd', 'lsqr', 'eigen')},
                              'score': 0.0}]

    with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_validation = data['X_validation']
        y_validation = data['y_validation']

    for candidate in candidate_classifiers:
        print(candidate['name'])

        candidate['classifier'].fit(X_train, y_train)
        predictions = candidate['classifier'].predict(X_validation)
        candidate['score'] = roc_auc_score(y_validation, predictions)

        print("----------------------------------------------------------------------")
        print("Score: ", candidate['score'])
        print("Accuracy: ", candidate['classifier'].score(X_validation, y_validation))
        print("F1 score: ", f1_score(y_validation, predictions, pos_label=1.0))
        print("----------------------------------------------------------------------\n\n")

    top_3_classifiers = sorted(candidate_classifiers, key=itemgetter('score'), reverse=True)[:3]
    pprint(top_3_classifiers[0])
    pprint(top_3_classifiers[1])
    pprint(top_3_classifiers[2])

    with open("best_candidates.p", "wb") as f:
        pickle.dump(top_3_classifiers, f)


if __name__ == "__main__":
    train_candidates()
