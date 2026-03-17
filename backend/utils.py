import torch
import os


# -------------------------------
# SAVE MODEL (WEIGHTS ONLY)
# -------------------------------
def save_model(model, path):
    """
    Save only model weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")


# -------------------------------
# LOAD MODEL (SAFE)
# -------------------------------
def load_model(model, path, device):
    """
    Load model weights safely on any device
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from {path}")

    return model


# -------------------------------
# SAVE FULL CHECKPOINT
# -------------------------------
def save_checkpoint(model, optimizer, epoch, best_iou, path):
    """
    Save full training checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_iou": best_iou
    }, path)

    print(f"✅ Checkpoint saved to {path}")


# -------------------------------
# LOAD FULL CHECKPOINT
# -------------------------------
def load_checkpoint(model, optimizer, path, device):
    """
    Load checkpoint (resume training)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint["epoch"]
    best_iou = checkpoint.get("best_iou", 0)

    model.to(device)

    print(f"✅ Checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch}")

    return model, optimizer, epoch, best_iou