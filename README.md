# 🧠 House Price Prediction - Neural Network

## Overview
This project uses a **PyTorch Neural Network** to predict California housing prices.

## 🏆 Results
- **R² Score:** 0.8344 (Beats XGBoost!)
- **RMSE:** $0.46K
- **Architecture:** 8 → 64 → 32 → 16 → 1

## 📊 Model Comparison
| Model | R² Score |
|-------|----------|
| Linear Regression | 0.5758 |
| Random Forest | 0.7739 |
| XGBoost | 0.8262 |
| **Neural Network** | **0.8344** 🏆 |

## 🚀 Deployment
- **FastAPI** for predictions
- **Streamlit** for interactive dashboard

## 📁 Structure
```
dl-house-prices/
├── src/
│   └── model.py                 # Neural Network definition
├── deployment/
│   ├── app.py                   # FastAPI
│   └── dashboard.py             # Streamlit
├── models/                       # Saved models
├── train.py                      # Training script
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
