# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === Paths ===
DATA_PATH = Path("data") / "heart.csv"
PLOTS_DIR = Path("plots") / "eda"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Cargar dataset ===
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded for EDA:", df.shape)

# === 1. Distribución de la variable objetivo ===
plt.figure(figsize=(6, 4))
sns.countplot(x="HeartDisease", data=df)
plt.title("Target Distribution (HeartDisease)")
plt.savefig(PLOTS_DIR / "target_distribution.png")
plt.close()

# === 2. Histograma de Edad por clase ===
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="HeartDisease", kde=True, bins=30)
plt.title("Age Distribution by Heart Disease")
plt.savefig(PLOTS_DIR / "age_distribution_by_target.png")
plt.close()

# === 3. Boxplot de colesterol por clase ===
if "Cholesterol" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="HeartDisease", y="Cholesterol", data=df)
    plt.title("Cholesterol Boxplot by Heart Disease")
    plt.savefig(PLOTS_DIR / "cholesterol_boxplot.png")
    plt.close()

# === 4. Mapa de correlación ===
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
plt.close()

# === 5. Gráficos de barras para variables categóricas ===
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=col, y="HeartDisease", data=df, estimator=lambda x: sum(x)/len(x))
        plt.title(f"{col} vs Heart Disease (proportion)")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.savefig(PLOTS_DIR / f"categorical_barplot_{col}.png")
        plt.close()

print(f"📊 EDA completed. Plots saved to: {PLOTS_DIR}")
