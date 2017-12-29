import pickle
from pprint import pprint

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV


def train_best():
    with open("best_candidates.p", "rb") as f:
        top_3_classifiers = pickle.load(f)

    with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_validation = data['X_validation']
        y_validation = data['y_validation']

    scorer = make_scorer(roc_auc_score)

    best_classifier = None
    best_score = float("-inf")

    for candidate in top_3_classifiers:
        print("Training a", candidate['name'], "classifier.")

        grid = GridSearchCV(candidate['classifier'], candidate['grid'], scorer)
        grid = grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        best_parameters = best_estimator.get_params()

        predictions = best_estimator.predict(X_validation)
        score = roc_auc_score(y_validation, predictions)

        if score > best_score:
            best_classifier = best_estimator
            best_score = score

        print("BEST PARAMETERS:")
        pprint(best_parameters)
        print("Tuned model has AUROC: ", score)
        print("---------------------- \n\n")

    with open("best.p", "wb") as f:
        pickle.dump(best_classifier, f)


if __name__ == "__main__":
    train_best()
