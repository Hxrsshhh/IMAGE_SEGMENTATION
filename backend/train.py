import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import SegmentationDataset
from models.model import get_model

torch.backends.cudnn.benchmark = True

# -------------------------------
# CONFIG
# -------------------------------
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"

VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 40
IGNORE_INDEX = 255

CHECKPOINT_PATH = "checkpoint.pth"


# -------------------------------
# IoU FUNCTION
# -------------------------------
def compute_iou(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if ious else 0


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    use_amp = device == "cuda"

    # ---------------- DATASET ----------------
    train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=use_amp
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=use_amp
    )

    # ---------------- MODEL ----------------
    model = get_model(NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_iou = 0
    start_epoch = 0

    train_losses = []
    val_losses = []

    # ---------------- LOAD CHECKPOINT ----------------
    if os.path.exists(CHECKPOINT_PATH):

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_iou = checkpoint.get("best_val_iou", 0)

        print(f"Resuming training from epoch {start_epoch}")

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        model.train()
        train_loss = 0

        loop = tqdm(train_loader)

        for images, masks, _ in loop:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(images)
                    loss = loss_fn(preds, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                preds = model(images)
                loss = loss_fn(preds, masks)

                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        print("Train Loss:", train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():

            for images, masks, _ in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)

                loss = loss_fn(preds, masks)
                val_loss += loss.item()

                val_iou += compute_iou(preds, masks, NUM_CLASSES)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        val_losses.append(val_loss)

        print("Validation Loss:", val_loss)
        print("Validation IoU:", val_iou)

        # ---------------- SAVE BEST MODEL (IoU) ----------------
        if val_iou > best_val_iou:

            best_val_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")

            print("🔥 Best model saved (based on IoU)!")

        # ---------------- SAVE CHECKPOINT ----------------
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_iou": best_val_iou
        }, CHECKPOINT_PATH)

        print("Checkpoint saved")

    # ---------------- SAVE LOSSES ----------------
    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)

    print("Training finished")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    train()