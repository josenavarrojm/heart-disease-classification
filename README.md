# 🫀 Heart Disease Classification

This project builds a complete pipeline for predicting heart disease using supervised machine learning techniques. The dataset used comes from Kaggle and includes medical data of patients along with their heart disease diagnosis. The goal is to create a reproducible, well-tested project that includes data preprocessing, model training and evaluation, prediction capabilities, and performance visualization.

---

## 📁 Project Structure

```
heart-disease-classification/
├── data/                   # Raw and processed data
│   ├── heart.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── scaler.pkl
│
├── models/                 # Trained models and performance logs
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   ├── model_performance.csv
│   ├── LogisticRegression_confusion_matrix.png
│   ├── LogisticRegression_roc_curve.png
│   ├── RandomForest_confusion_matrix.png
│   └── RandomForest_roc_curve.png
│
├── plots/                  # Visualizations
│   └── evaluation/         # Confusion matrices and ROC curves
│
├── notebooks/              # Exploratory analysis and results
│   ├── EDA.ipynb
│   └── Model_Results.ipynb
│
├── src/                    # Core scripts
│   ├── data_prep.py        # Data download, split, and preprocessing
│   ├── train_model.py      # Training and hyperparameter tuning
│   ├── evaluate.py         # Model evaluation and metrics visualization
│   ├── predict.py          # Inference with manual or CSV input
│   └── eda.py              # (Optional) Data visualizations
│
├── tests/                  # Unit tests
│   ├── test_data_prep.py
│   ├── test_train_model.py
│   ├── test_predict.py
│   └── test_evaluate.py
│
├── .gitignore
├── requirements.txt
├── run_pipeline.sh         # Bash script to run the full pipeline
├── setup.py                # Python project configuration
└── README.md               # Project documentation
```

---

## 🚀 Quickstart

```bash
# 1. Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
bash run_pipeline.sh

# 4. Run unit tests
pytest tests/
```

---

## 📊 Features

- Downloads and prepares data from Kaggle
- Preprocesses features including one-hot encoding and scaling
- Trains two models with hyperparameter tuning using GridSearchCV:
  - Logistic Regression
  - Random Forest
- Saves models and scalers
- Generates classification reports, confusion matrices, and ROC curves
- Makes predictions via CLI or CSV input
- Includes unit tests for key components
- Includes two Jupyter notebooks for:
  - Exploratory Data Analysis (EDA)
  - Visualizing and comparing model performance

---

## 📈 Results

All models were trained and evaluated using a 80/20 train-test split. Both Logistic Regression and Random Forest achieved strong F1 scores:

| Model              | Accuracy | F1 Score | AUC  |
|-------------------|----------|----------|------|
| LogisticRegression| 0.89     | 0.90     | 0.91 |
| RandomForest      | 0.89     | 0.90     | 0.91 |

Performance graphs are saved in the `/plots/evaluation/` directory.

---

## 🔍 Dataset

Dataset: [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
Author: Federico Soriano  
License: Open Data / Public Domain

---

## 🧪 Testing

Run all tests with:
```bash
pytest tests/
```

All critical components including preprocessing, training, prediction, and evaluation are tested.

---

## ⚙️ Setup and Installation

This repository includes a `setup.py` file for package-style installation. You can install it locally with:
```bash
pip install .
```

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 👨‍💻 Author
 
**José Junior Navarro Meneses**  
Electronics Engineer | Data Scientist | Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/jose-junior-navarro-meneses-913b62338/) · [GitHub](https://github.com/josenavarrojm)
Email: josenavarrojmx@gmail.com

---

## ✅ Future Work (Optional Enhancements)
- Add XGBoost and SVM models
- Add SHAP explanations for model interpretability
- Deploy the model as a REST API (FastAPI)
- Dockerize the entire pipeline


