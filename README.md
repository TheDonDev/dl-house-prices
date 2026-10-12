# 🧠 Nairobi Property Price Predictor

## Project Description
This project is a machine learning application for predicting Nairobi property prices using a neural network built with PyTorch. It combines data preprocessing, model training, and deployment into a simple end-to-end workflow for evaluating price trends and comparing model performance.

## Overview
This project uses a **PyTorch Neural Network** to predict Nairobi property prices from property characteristics such as bedrooms, bathrooms, house size, and land size.

## 🏆 Results
- **R² Score:** Model performance depends on the trained Nairobi dataset.
- **Architecture:** 4 → 64 → 32 → 16 → 1

## 📊 Model Inputs
- **Bedrooms**
- **Bathrooms**
- **House size (m²)**
- **Land size (acres)**

## 🚀 Deployment
- **FastAPI** for predictions
- **Streamlit** for interactive dashboard

## 📁 Structure
```
ml-property-prices/
├── src/
│   └── model.py                 # Neural Network definition
├── deployment/
│   ├── app.py                   # FastAPI
│   └── dashboard.py             # Streamlit
├── models/                      # Saved models
├── train.py                     # Training script
├── Nairobi propertyprices - Sheet1.csv  # Nairobi dataset
├── README.md
└── requirements.txt
```

## 🔧 Run
```bash
# Train
python train.py

# API
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# Dashboard
streamlit run deployment/dashboard.py
```
