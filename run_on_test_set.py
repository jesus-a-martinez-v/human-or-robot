import pickle


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


if __name__ == "__main__":
    run_on_test_set()
