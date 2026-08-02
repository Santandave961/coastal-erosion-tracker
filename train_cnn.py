"""
train_cnn.py - Train a CNN to classify coastal erosion tiles (eroded vs stable).

Assumes preprocessed .npy tiles (H, W, 3) from preprocess_tile.py, labeled via
a CSV: filename,label   where label is 0 (stable) or 1 (eroded).

Built for small/growing label sets (start with a handful, add more as you label
more tiles) -- uses stratified split, class weighting, and early stopping so it
doesn't fall apart with <50 examples.

Usage:
    python train_cnn.py --labels labels.csv --tiles_dir preprocessed/ --epochs 30

labels.csv format:
    filename,label
    response_0001.npy,0
    response_0002.npy,1
    ...
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


class TileDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx]).astype(np.float32)  # (H, W, C)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)        # (C, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def load_labels(labels_csv, tiles_dir):
    filepaths, labels = [], []
    # utf-8-sig strips a BOM if present (common when a CSV is saved from Excel);
    # it's a no-op if there's no BOM, so this is safe either way.
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # normalize header names: strip whitespace, lowercase
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            fp = Path(tiles_dir) / row["filename"]
            if not fp.exists():
                print(f"  WARNING: missing file, skipping: {fp}")
                continue
            filepaths.append(str(fp))
            labels.append(int(row["label"]))
    return filepaths, np.array(labels)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    filepaths, labels = load_labels(args.labels, args.tiles_dir)
    print(f"Loaded {len(filepaths)} labeled tiles "
          f"({(labels == 1).sum()} eroded / {(labels == 0).sum()} stable)")

    if len(filepaths) < 10:
        print("WARNING: very small dataset. Results will be noisy until you label more tiles.")

    stratify = labels if len(set(labels)) > 1 and min(np.bincount(labels)) >= 2 else None
    train_fp, val_fp, train_y, val_y = train_test_split(
        filepaths, labels, test_size=args.val_split, random_state=42, stratify=stratify
    )

    train_ds = TileDataset(train_fp, train_y)
    val_ds = TileDataset(val_fp, val_y)

    # Weighted sampler to handle class imbalance (common with rare erosion events)
    class_counts = np.bincount(train_y, minlength=2)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[train_y]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(in_channels=args.in_channels).to(device)
    pos_weight = torch.tensor([class_weights[0] / max(class_weights[1], 1e-6)]).to(device) \
        if len(class_counts) > 1 and class_counts[1] > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(args.out) / "best_model.pt"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        try:
            auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_auc={auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                break

    print(f"\nBest model saved to: {ckpt_path}")

    # Final report on val set using best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(y.numpy().astype(int))

    print("\n--- Validation report (best checkpoint) ---")
    print(classification_report(all_labels, all_preds, target_names=["stable", "eroded"], zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to labels.csv (filename,label)")
    parser.add_argument("--tiles_dir", required=True, help="Directory containing preprocessed .npy tiles")
    parser.add_argument("--out", default="checkpoints", help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    args = parser.parse_args()
    train(args)