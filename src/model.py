"""
Neural Network for Nairobi House Price Prediction using PyTorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class HousePriceNN(nn.Module):
    """
    Neural Network architecture:
    - Input: 8 features
    - Hidden layers: 64 → 32 → 16 neurons
    - Output: 1 (price prediction)
    """

    def __init__(self, input_size=8):
        super(HousePriceNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def train_model(model, train_loader, test_loader, epochs=200, lr=0.001):
    """
    Train the neural network
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 Training Nairobi House Price Neural Network...")
    print("=" * 50)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    print("=" * 50)
    print("✅ Training complete!")
    return model


if __name__ == "__main__":
    # Quick test
    model = HousePriceNN()
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(model)
