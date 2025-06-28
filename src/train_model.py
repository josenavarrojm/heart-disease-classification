import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib

# === Rutas ===
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# === Cargar datos preprocesados ===
with open(DATA_DIR / "X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(DATA_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(DATA_DIR / "y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open(DATA_DIR / "y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# === Inicializar modelos y grids ===

models = {
    "LogisticRegression": (
        LogisticRegression(solver="liblinear", random_state=42),
        {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    )
}

# === Entrenamiento con GridSearchCV ===
best_models = []  # <- almacenamos modelos y nombres
performance = []  # <- para métricas completas

for name, (model, param_grid) in models.items():
    print(f"\n🔍 Training and tuning {name}...")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # Mejor modelo y predicciones
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Métricas
    performance.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba)
    })

    print(f"✅ Best {name}: {grid.best_params_}")
    print(f"🎯 Accuracy: {performance[-1]['Accuracy']:.4f} | F1 Score: {performance[-1]['F1 Score']:.4f}")

    # Guardar modelo
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Model saved to {model_path}")

# === Guardar CSV con todas las métricas ===
df_perf = pd.DataFrame(performance)
df_perf.to_csv(MODELS_DIR / "model_performance.csv", index=False)
print("📄 model_performance.csv saved.")
