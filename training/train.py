import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.unet import UNet
from training.dataset import MicroscopyDataset
from training.losses import CombinedLoss

DATA_BASE = r"D:\Projects\Microscopy_AI_Project\dataset\degraded"
BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 1e-4
PATCH_SIZE = 256
CHECKPOINT_DIR = r"D:\Projects\Microscopy_AI_Project\results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = MicroscopyDataset(
    clean_dir=os.path.join(DATA_BASE, "train", "clean"),
    degraded_dir=os.path.join(DATA_BASE, "train", "degraded"),
    patch_size=PATCH_SIZE
)
val_dataset = MicroscopyDataset(
    clean_dir=os.path.join(DATA_BASE, "val", "clean"),
    degraded_dir=os.path.join(DATA_BASE, "val", "degraded"),
    patch_size=PATCH_SIZE
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")

model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
criterion = CombinedLoss(l1_weight=0.5, ssim_weight=0.3, edge_weight=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for degraded, clean in train_loader:
        degraded, clean = degraded.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(degraded)
        loss, l1_val, ssim_val = criterion(output, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for degraded, clean in val_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            output = model(degraded)
            loss, l1_val, ssim_val = criterion(output, clean)
            val_running_loss += loss.item()
    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0:
        print(f"  Current learning rate: {current_lr:.2e}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"  -> New best model saved (val loss {avg_val_loss:.4f})")

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
print("Training complete. Loss curve saved to results/loss_curve.png")