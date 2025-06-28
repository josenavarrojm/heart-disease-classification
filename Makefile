# Makefile - Heart Disease Classification
# Autor: Jose Navarro Meneses

# Python executable
PYTHON = .venv/Scripts/python.exe

# Rutas
SRC_DIR = src
DATA_DIR = data
MODELS_DIR = models
PLOTS_DIR = plots

# Tarea principal
all: clean prepare train evaluate eda predict

# === TAREAS ===

# Preparación de datos
prepare:
	@echo "🔧 Preprocessing data..."
	$(PYTHON) $(SRC_DIR)/data_prep.py

# Entrenamiento
train:
	@echo "🚀 Training models..."
	$(PYTHON) $(SRC_DIR)/train_model.py

# Evaluación
evaluate:
	@echo "📊 Evaluating models..."
	$(PYTHON) $(SRC_DIR)/evaluate.py

# Análisis exploratorio
eda:
	@echo "🔎 Running EDA..."
	$(PYTHON) $(SRC_DIR)/eda.py

# Predicción con valores de prueba
predict:
	@echo "🧪 Predicting example manually..."
	$(PYTHON) $(SRC_DIR)/predict.py --manual

# Limpiar archivos de salida
clean:
	@echo "🧹 Cleaning output folders..."
	del /Q $(DATA_DIR)\*.pkl 2>nul || true
	del /Q $(MODELS_DIR)\*.pkl 2>nul || true
	del /Q $(MODELS_DIR)\*.csv 2>nul || true
	del /Q $(PLOTS_DIR)\evaluation\*.png 2>nul || true
	del /Q $(PLOTS_DIR)\eda\*.png 2>nul || true
