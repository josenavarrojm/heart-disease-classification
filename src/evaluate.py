# src/evaluate.py

import pickle
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# === Rutas ===
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots") / "evaluation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Cargar datos ===
with open(DATA_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(DATA_DIR / "y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# === Lista de modelos a evaluar ===
model_files = ["LogisticRegression.pkl", "RandomForest.pkl"]

for model_file in model_files:
    model_path = MODELS_DIR / model_file
    model_name = model_path.stem

    print(f"\n📊 Evaluating model: {model_name}")

    # === Cargar modelo ===
    model = joblib.load(model_path)

    # === Predicciones ===
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # === Reporte de clasificación ===
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # === Matriz de confusión ===
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png")
    plt.close()

    # === Curva ROC ===
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(PLOTS_DIR / f"{model_name}_roc_curve.png")
    plt.close()

    print(f"✅ Saved confusion matrix and ROC curve for {model_name} in {PLOTS_DIR}")
