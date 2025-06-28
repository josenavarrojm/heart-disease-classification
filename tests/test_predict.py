# tests/test_predict.py

import pickle
import joblib
import numpy as np

def test_prediction_output():
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    model = joblib.load("models/LogisticRegression.pkl")

    y_pred = model.predict(X_test[:5])
    assert len(y_pred) == 5
    assert set(np.unique(y_pred)).issubset({0, 1})
