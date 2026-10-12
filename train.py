"""
Train a neural network for Nairobi property price prediction.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.model import HousePriceNN, train_model

DATA_PATH = Path(__file__).resolve().parent / "Nairobi propertyprices - Sheet1.csv"
FEATURE_COLUMNS = ["Bedroom", "bathroom", "House size", "Land size"]


def clean_numeric(value):
    """Convert values like 'KSh 35 000 000' and '230 m²' into floats."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan

    text = text.replace("KSh", "").replace("ksh", "").replace("Ksh", "")
    text = text.replace("KES", "").replace("kes", "")
    text = text.replace("Sh", "")
    text = text.replace(",", "").replace(" ", "")
    text = text.replace("m²", "").replace("M²", "").replace("m2", "").replace("M2", "")
    text = text.replace("acres", "").replace("acre", "")
    text = text.replace("²", "")

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan

    return float(numbers[0]) if len(numbers) == 1 else float("".join(numbers))


def load_nairobi_dataset():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    df["Price"] = df["Price"].apply(clean_numeric)
    for column in FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric)

    df = df.dropna(subset=["Price"] + FEATURE_COLUMNS, how="any").copy()
    df["Price"] = df["Price"].astype(float)

    for column in FEATURE_COLUMNS:
        df[column] = df[column].astype(float)

    return df


def main():
    print("🏠 Nairobi Property Price Prediction - Neural Network")
    print("=" * 60)

    df = load_nairobi_dataset()
    X = df[FEATURE_COLUMNS].values
    y = df["Price"].values.reshape(-1, 1)

    print(f"📊 Dataset: {len(df)} samples, {X.shape[1]} features")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    joblib.dump(imputer, "models/imputer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = HousePriceNN(input_size=X.shape[1])
    print(f"\n🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    model = train_model(model, train_loader, test_loader, epochs=250, lr=0.001)

    torch.save(model.state_dict(), "models/model.pth")
    joblib.dump(model, "models/model.pkl")
    print("💾 Models saved to models/")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n📊 Neural Network Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: KSh {rmse:,.0f}")

    if r2 > 0.8:
        print("   🎉 Neural Network performs well on Nairobi property data!")
    else:
        print("   💡 Consider collecting more Nairobi listings or tuning the model")


if __name__ == "__main__":
    main()
