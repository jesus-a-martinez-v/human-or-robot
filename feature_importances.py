import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances():
    with open("train.p", "rb") as f:
        data = pickle.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_validation = data['X_validation']
        y_validation = data['y_validation']

    with open("best.p", "rb") as f:
        best_classifier = pickle.load(f)

    importances = best_classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in best_classifier.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_validation.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_validation.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_validation.shape[1]), indices)
    plt.xlim([-1, X_validation.shape[1]])
    # plt.show()
    plt.savefig("feature_importances.png")
    plt.close()


if __name__ == "__main__":
    plot_feature_importances()
