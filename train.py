"""
Train Neural Network for House Price Prediction
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from src.model import HousePriceNN, train_model

def main():
    print("🏠 House Price Prediction - Neural Network")
    print("=" * 60)
    
    # 1. Load data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 4. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    # 5. Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📦 Batch size: {batch_size}")
    print(f"📦 Training batches: {len(train_loader)}")
    print(f"📦 Test batches: {len(test_loader)}")
    
    # 6. Create and train model
    model = HousePriceNN()
    print(f"\n🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model = train_model(model, train_loader, test_loader, epochs=200, lr=0.001)
    
    # 7. Save model
    torch.save(model.state_dict(), 'models/model.pth')
    joblib.dump(model, 'models/model.pkl')
    print("💾 Models saved to models/")
    
    # 8. Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\n📊 Neural Network Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:.2f}K")
    
    # 9. Compare with XGBoost
    print("\n📈 Model Comparison:")
    print(f"   XGBoost R²: 0.8262")
    print(f"   Neural Net R²: {r2:.4f}")
    
    if r2 > 0.8262:
        print("   🎉 Neural Network performs better!")
    else:
        print("   💡 XGBoost still better - try more epochs or different architecture")

if __name__ == "__main__":
    main()
