# tests/test_evaluate_outputs.py

from pathlib import Path

def test_evaluation_graphics_exist():
    models = ["LogisticRegression", "RandomForest"]
    plot_dir = Path("plots/evaluation")

    for model in models:
        cm_file = plot_dir / f"{model}_confusion_matrix.png"
        roc_file = plot_dir / f"{model}_roc_curve.png"

        assert cm_file.exists(), f"{cm_file} not found"
        assert roc_file.exists(), f"{roc_file} not found"

def test_metrics_csv_exists():
    metrics_file = Path("models/model_performance.csv")
    assert metrics_file.exists(), "model_performance.csv not found"
