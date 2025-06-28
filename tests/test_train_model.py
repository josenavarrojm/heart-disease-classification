from pathlib import Path

def test_model_files_exist():
    models = ["LogisticRegression.pkl", "RandomForest.pkl"]
    for model in models:
        assert (Path("models") / model).exists(), f"{model} not found"
