import pickle
from pathlib import Path

def test_preprocessed_files_exist():
    expected_files = [
        "X_train.pkl", "X_test.pkl",
        "y_train.pkl", "y_test.pkl",
        "scaler.pkl"
    ]
    for filename in expected_files:
        assert (Path("data") / filename).exists(), f"{filename} not found"

def test_shapes():
    X_train = pickle.load(open("data/X_train.pkl", "rb"))
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    y_train = pickle.load(open("data/y_train.pkl", "rb"))
    y_test = pickle.load(open("data/y_test.pkl", "rb"))

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
