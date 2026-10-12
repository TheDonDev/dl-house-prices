"""
FastAPI for Nairobi House Price Prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HousePriceNN

app = FastAPI(
    title="Nairobi House Price Prediction - Neural Network",
    description="Predict Nairobi housing prices using a Neural Network",
    version="2.0.0"
)

# Load model
model = HousePriceNN()
model_path = Path(__file__).parent.parent / 'models' / 'model.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Load scaler
scaler_path = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
scaler = joblib.load(scaler_path)

FEATURE_NAMES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def root():
    return {
        "message": "🏠 Nairobi House Price Prediction - Neural Network",
        "model": "Neural Network (PyTorch)",
        "features": FEATURE_NAMES
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        X = np.array([[
            features.MedInc, features.HouseAge, features.AveRooms,
            features.AveBedrms, features.Population, features.AveOccup,
            features.Latitude, features.Longitude
        ]])

        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(X_tensor).numpy()[0][0]

        return {
            "prediction": float(prediction),
            "price_k": f"${prediction:.2f}K"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Neural Network"}
