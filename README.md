
# 🫀 Heart Disease Classification

This project aims to build a classification model to predict the presence of heart disease using the `heart.csv` dataset. It involves preprocessing techniques, model training, and performance evaluation, all with a focus on reproducibility and production-readiness.

## 📁 Project Structure

```
heart-disease-classification/
│
├── data/                    # Raw and processed data
│   ├── heart.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── scaler.pkl
│
├── models/                  # Trained models and evaluation metrics
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   └── model_performance.csv
│
├── .venv/                   # Python virtual environment
│
├── data_prep.py            # Data preprocessing script
├── train_model.py          # Model training and evaluation script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 📌 Objective

To develop a binary classification system that predicts whether a patient has heart disease based on clinical features.

## 🧪 Technologies Used

- Python 3.x
- scikit-learn
- pandas
- matplotlib / seaborn
- joblib / pickle
- VSCode

## ⚙️ Usage Instructions

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the data:
   ```bash
   python data_prep.py
   ```

4. Train the models:
   ```bash
   python train_model.py
   ```

## 📊 Results

The `train_model.py` script generates performance metrics such as accuracy, precision, recall, and F1-score, and saves them to `models/model_performance.csv`.

| Model              | Accuracy | Precision | Recall | F1-score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression|    ...   |    ...    |  ...   |   ...    |
| RandomForest      |    ...   |    ...    |  ...   |   ...    |

> These values are populated after running the training script.

## 📈 Future Work

- Comparison with other algorithms (XGBoost, SVM, etc.)
- Visualizations (confusion matrix, ROC curve)
- Hyperparameter tuning
- Exporting to ONNX or joblib for deployment

## 🧑‍💻 Author

**José Junior Navarro Meneses**  
Electronics Engineer | Data Scientist | Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/)

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
