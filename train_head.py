# train_head.py
# ============================
# Train Multi-Feature Tension Head
# ============================

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import TensionClassifier


FEATURES = [
    "avg_tension",
    "max_tension",
    "std_tension",
    "slope",
    "early_late_ratio",
    "num_spikes"
]


df = pd.read_csv("train_tension.csv")

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

LABEL_MAP = {
    "consistent": 1,
    "contradict": 0,
    "1": 1,
    "0": 0
}

df["label"] = df["label"].map(LABEL_MAP)

if df["label"].isnull().any():
    raise ValueError("❌ Unknown labels found")

X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.float32).unsqueeze(1)

model = TensionClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 300

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "tension_head.pt")
print("✅ tension_head.pt saved")
