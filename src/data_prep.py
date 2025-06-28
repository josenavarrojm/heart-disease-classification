import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

# === Paths ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# === 1. Load dataset from Kaggle ===

csv_path = DATA_DIR / "heart.csv"

if csv_path.exists():
    print(f"Loading existing dataset from {csv_path}")
    df = pd.read_csv(csv_path)
else:
    print("Downloading dataset from Kaggle...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/heart-failure-prediction",
        "heart.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")
# === 2. Save original dataset ===
    df.to_csv(csv_path, index=False)

# print(df.head(4))

# === Clean / Encode categorical columns ===
# print("Initial columns and types:\n", df.dtypes)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

if categorical_cols:
    # print("Encoding categorical columns:", categorical_cols)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    print("No categorical columns found.")


# === 3. Separate features and labels ===
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === 5. Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Save processed data ===
with open(DATA_DIR / "X_train.pkl", "wb") as f:
    pickle.dump(X_train_scaled, f)
with open(DATA_DIR / "X_test.pkl", "wb") as f:
    pickle.dump(X_test_scaled, f)
with open(DATA_DIR / "y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open(DATA_DIR / "y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open(DATA_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Preprocessed data saved to /data/")