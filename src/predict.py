import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import argparse
import sys

# === Rutas ===
DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# === Cargar modelo y scaler ===
model_path = MODELS_DIR / "RandomForest.pkl"  # o LogisticRegression.pkl
scaler_path = DATA_DIR / "scaler.pkl"

model = joblib.load(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# === Argument parser ===
parser = argparse.ArgumentParser(description="Heart disease prediction")
parser.add_argument("--csv", type=str, help="Ruta al archivo CSV con nuevos pacientes")

args = parser.parse_args()

# === Función para predecir ===
def predict_from_dataframe(df):
    print("\n🧪 Original input:")
    print(df)

    # Escalado
    X = scaler.transform(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Mostrar resultados
    results = df.copy()
    results["Prediction"] = preds
    results["Probability"] = probs

    print("\n✅ Prediction Results:")
    print(results[["Prediction", "Probability"]])
    return results

# === CSV mode ===
if args.csv:
    try:
        new_data = pd.read_csv(args.csv)
        predict_from_dataframe(new_data)
    except Exception as e:
        print("❌ Error loading CSV:", e)
        sys.exit(1)

# === Manual mode ===
else:
    print("📝 Manual input mode:")
    print("ℹ️ Ingrese valores numéricos para variables continuas, y 0 o 1 para variables booleanas.")
    print("")

    input_features = [
        "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
        "Sex_M", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
        "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
        "ST_Slope_Flat", "ST_Slope_Up"
    ]

    boolean_features = {
        feat for feat in input_features
        if "_" in feat  # asumen codificación one-hot para categorías
    }

    input_values = []

    for feat in input_features:
        while True:
            val = input(f"{feat} ({'0/1' if feat in boolean_features else 'numeric'}): ").strip()
            try:
                if feat in boolean_features:
                    val = int(val)
                    if val not in (0, 1):
                        raise ValueError()
                    input_values.append(bool(val))
                else:
                    input_values.append(float(val))
                break
            except ValueError:
                print(f"❌ Valor inválido para {feat}. Intente de nuevo.")

    df_manual = pd.DataFrame([input_values], columns=input_features)
    predict_from_dataframe(df_manual)

