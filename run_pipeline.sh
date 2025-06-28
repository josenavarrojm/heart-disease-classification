#!/bin/bash

# =====================================
# run_pipeline.sh
# Autor: Jose Navarro Meneses
# Descripción: Ejecuta el pipeline completo del proyecto
# =====================================

# Colores para terminal
GREEN='\033[0;32m'
NC='\033[0m' # sin color

PYTHON=".venv/Scripts/python.exe"
SRC="src"

echo -e "${GREEN}🔧 Step 1: Preprocessing data...${NC}"
$PYTHON $SRC/data_prep.py || exit 1

echo -e "${GREEN}🚀 Step 2: Training models...${NC}"
$PYTHON $SRC/train_model.py || exit 1

echo -e "${GREEN}📊 Step 3: Evaluating models...${NC}"
$PYTHON $SRC/evaluate.py || exit 1

echo -e "${GREEN}🔍 Step 4: Running EDA...${NC}"
$PYTHON $SRC/eda.py || exit 1

echo -e "${GREEN}🧪 Step 5: Predicting sample...${NC}"
$PYTHON $SRC/predict.py --manual || exit 1

echo -e "${GREEN}✅ Pipeline completed successfully.${NC}"
