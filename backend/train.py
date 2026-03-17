import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import SegmentationDataset
from models.model import get_model

torch.backends.cudnn.benchmark = True

# dataset paths
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"

VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 40

CHECKPOINT_PATH = "checkpoint.pth"


def train():

    # ---------------- DEVICE ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    use_amp = device == "cuda"

    # ---------------- LOAD DATASET ----------------
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
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    start_epoch = 0

    # ---------------- LOAD CHECKPOINT ----------------
    if os.path.exists(CHECKPOINT_PATH):

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resuming training from epoch {start_epoch}")

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(start_epoch, EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        model.train()

        train_loss = 0
        loop = tqdm(train_loader)

        for images, masks in loop:

            images = images.to(device)
            masks = masks.long().to(device)

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

        train_loss = train_loss / len(train_loader)

        print("Train Loss:", train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.long().to(device)

                preds = model(images)

                loss = loss_fn(preds, masks)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        print("Validation Loss:", val_loss)

        # ---------------- SAVE BEST MODEL ----------------
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

            print("Best model saved!")

        # ---------------- SAVE CHECKPOINT ----------------
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, CHECKPOINT_PATH)

        print("Checkpoint saved")

    print("Training finished")


if __name__ == "__main__":
    train()