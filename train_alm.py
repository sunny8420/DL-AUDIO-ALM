import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import random

# Load features
speech_X = np.load("model/features/speech_X.npy", allow_pickle=True)
speech_y = np.load("model/features/speech_y.npy", allow_pickle=True)
sound_X = np.load("model/features/sound_X.npy", allow_pickle=True)
sound_y = np.load("model/features/sound_y.npy", allow_pickle=True)

# Encode labels
speech_enc = LabelEncoder()
sound_enc = LabelEncoder()

speech_y = speech_enc.fit_transform(speech_y)
sound_y = sound_enc.fit_transform(sound_y)

# Dataset classes
class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SoundDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

speech_ds = SpeechDataset(speech_X, speech_y)
sound_ds = SoundDataset(sound_X, sound_y)

speech_loader = DataLoader(speech_ds, batch_size=32, shuffle=True)
sound_loader = DataLoader(sound_ds, batch_size=32, shuffle=True)

# Model
class ALMNet(nn.Module):
    def __init__(self, input_size, speech_classes, sound_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.speech_head = nn.Linear(128, speech_classes)
        self.sound_head = nn.Linear(128, sound_classes)

    def forward(self, x):
        z = self.shared(x)
        return self.speech_head(z), self.sound_head(z)

model = ALMNet(
    speech_X.shape[1],
    len(set(speech_y)),
    len(set(sound_y))
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training
EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0

    for (sx, sy), (nx, ny) in zip(speech_loader, sound_loader):
        optimizer.zero_grad()

        outS, _ = model(sx)
        _, outN = model(nx)

        loss = loss_fn(outS, sy) + loss_fn(outN, ny)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/steps:.4f}")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "speech_encoder": speech_enc,
    "sound_encoder": sound_enc
}, "model/alm_model.pth")

print("\n🔥 MODEL TRAINED AND SAVED: model/alm_model.pth")
