"""
FastAPI for Nairobi property price prediction.
"""
from pathlib import Path
import os
import sys

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HousePriceNN

app = FastAPI(
    title="Nairobi Property Price Prediction - Neural Network",
    description="Predict Nairobi property prices using a Neural Network",
    version="2.0.0"
)

model = HousePriceNN(input_size=4)
model_path = Path(__file__).parent.parent / "models" / "model.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

imputer_path = Path(__file__).parent.parent / "models" / "imputer.pkl"
scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = ["bedrooms", "bathrooms", "house_size_m2", "land_size_acres"]


class PropertyFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    house_size_m2: float
    land_size_acres: float


@app.get("/")
def root():
    return {
        "message": "🏠 Nairobi Property Price Prediction - Neural Network",
        "model": "Neural Network (PyTorch)",
        "features": FEATURE_NAMES
    }


@app.post("/predict")
def predict(features: PropertyFeatures):
    try:
        values = np.array([[
            features.bedrooms,
            features.bathrooms,
            features.house_size_m2,
            features.land_size_acres,
        ]], dtype=float)

        values_imputed = imputer.transform(values)
        values_scaled = scaler.transform(values_imputed)
        X_tensor = torch.tensor(values_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(X_tensor).numpy()[0][0]

        return {
            "prediction": float(prediction),
            "price_ksh": f"KSh {prediction:,.0f}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Neural Network"}
