import torch


def save_model(model, path):
    """
    Save model weights
    """

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load model weights safely on any device
    """

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {path}")

    return model