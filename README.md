
# 🫀 Heart Disease Classification

Este proyecto consiste en construir un modelo de clasificación para predecir la presencia de enfermedades cardíacas utilizando el conjunto de datos `heart.csv`. Se han aplicado técnicas de preprocesamiento, entrenamiento de modelos y evaluación de desempeño, con el objetivo de desarrollar una solución reproducible y lista para producción.

## 📁 Estructura del Proyecto

```
heart-disease-classification/
│
├── data/                    # Datos originales y procesados
│   ├── heart.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── scaler.pkl
│
├── models/                  # Modelos entrenados y métricas
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   └── model_performance.csv
│
├── .venv/                   # Entorno virtual de Python
│
├── data_prep.py            # Script de preprocesamiento
├── train_model.py          # Script de entrenamiento y evaluación
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación del proyecto
```

## 📌 Objetivo

Desarrollar un sistema de clasificación binaria capaz de predecir si un paciente tiene una enfermedad cardíaca basada en características clínicas disponibles.

## 🧪 Tecnologías utilizadas

- Python 3.x
- scikit-learn
- pandas
- matplotlib / seaborn
- joblib / pickle
- VSCode

## ⚙️ Instrucciones de uso

1. Clonar el repositorio y crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # En Windows
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocesar los datos:
   ```bash
   python data_prep.py
   ```

4. Entrenar los modelos:
   ```bash
   python train_model.py
   ```

## 📊 Resultados

El script `train_model.py` genera métricas como precisión, recall y F1-score, y las guarda en `models/model_performance.csv`.

| Model              | Accuracy | Precision | Recall | F1-score |
|-------------------|----------|-----------|--------|----------|
| LogisticRegression|    ...   |    ...    |  ...   |   ...    |
| RandomForest      |    ...   |    ...    |  ...   |   ...    |

> Estos resultados se completan automáticamente tras ejecutar el entrenamiento.

## 📈 Futuro del proyecto

- Comparación con otros algoritmos (XGBoost, SVM, etc.)
- Visualizaciones (matriz de confusión, curva ROC)
- Optimización de hiperparámetros
- Exportación a formato ONNX o joblib para despliegue

## 🧑‍💻 Autor

**José Junior Navarro Meneses**  
Ingeniero Electrónico | Científico de Datos | Ingeniero de Machine Learning  
[LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
