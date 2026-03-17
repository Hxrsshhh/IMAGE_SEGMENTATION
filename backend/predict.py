import os
import cv2
import numpy as np
import torch

from models.model import get_model

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 40
IMG_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# COLORS (40 classes)
# -------------------------------
np.random.seed(42)  # for consistency
colors = np.random.randint(0, 255, (NUM_CLASSES, 3))


# -------------------------------
# MODEL
# -------------------------------
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    image = (image - 0.5) / 0.5
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image.to(device)


# -------------------------------
# PREDICT
# -------------------------------
def predict(image):
    image_tensor = preprocess(image)

    with torch.no_grad():
        pred = model(image_tensor)

    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    return mask


# -------------------------------
# COLOR MASK
# -------------------------------
def color_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(NUM_CLASSES):
        color[mask == c] = colors[c]

    return color


# -------------------------------
# OVERLAY
# -------------------------------
def overlay(image, mask, alpha=0.5):
    mask_colored = color_mask(mask)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)


# -------------------------------
# VISUALIZE
# -------------------------------
def visualize(image_path, gt_path, save_path):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # GT is already encoded (0–39)
    gt = cv2.imread(gt_path, 0)
    gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    pred_mask = predict(image_rgb)

    pred_color = color_mask(pred_mask)
    gt_color = color_mask(gt)
    overlay_img = overlay(image_rgb, pred_mask)

    combined = np.concatenate([
        image_rgb,
        gt_color,
        pred_color,
        overlay_img
    ], axis=1)

    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))