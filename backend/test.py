import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from dataset import SegmentationDataset
from models.model import get_model

# ---------------- CONFIG ----------------
TEST_IMG_DIR = "dataset/test/images"
TEST_MASK_DIR = "dataset/test/masks"

MODEL_PATH = "best_model.pth"
NUM_CLASSES = 40
IMG_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- COLORS (FOR 40 CLASSES) ----------------
np.random.seed(42)
colors = np.random.randint(0, 255, (NUM_CLASSES, 3))


# ---------------- FUNCTIONS ----------------
def color_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(NUM_CLASSES):
        color[mask == c] = colors[c]

    return color


def overlay(image, mask, alpha=0.5):
    mask_color = color_mask(mask)
    return cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)


def compute_iou(pred, gt):
    ious = []

    for cls in range(NUM_CLASSES):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if ious else 0


# ---------------- LOAD MODEL ----------------
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# ---------------- CHECK GT AVAILABILITY ----------------
has_gt = os.path.exists(TEST_MASK_DIR)

if has_gt:
    dataset = SegmentationDataset(TEST_IMG_DIR, TEST_MASK_DIR)
else:
    dataset = SegmentationDataset(TEST_IMG_DIR, TEST_IMG_DIR)  # dummy
    print("⚠️ No test masks found → running inference only")


# ---------------- CREATE OUTPUT FOLDERS ----------------
os.makedirs("outputs/comparisons", exist_ok=True)
os.makedirs("outputs/overlays", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/failures", exist_ok=True)


# ---------------- TEST LOOP ----------------
all_ious = []

for image, mask, name in tqdm(dataset):

    # Convert tensor → image
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = ((image_np * 0.5) + 0.5) * 255
    image_np = image_np.astype(np.uint8)

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)

    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    if has_gt:
        gt_mask = mask.numpy()
        iou = compute_iou(pred_mask, gt_mask)
        all_ious.append(iou)
    else:
        gt_mask = np.zeros_like(pred_mask)
        iou = -1

    # ---------------- VISUALS ----------------
    pred_color = color_mask(pred_mask)
    gt_color = color_mask(gt_mask)
    overlay_img = overlay(image_np, pred_mask)

    combined = np.concatenate([
        image_np,
        gt_color,
        pred_color,
        overlay_img
    ], axis=1)

    # ---------------- SAVE ----------------
    cv2.imwrite(f"outputs/comparisons/{name}", combined)
    cv2.imwrite(f"outputs/overlays/{name}", overlay_img)
    cv2.imwrite(f"outputs/predictions/{name}", pred_color)

    # ---------------- FAILURE ----------------
    if has_gt and iou < 0.3:
        cv2.imwrite(f"outputs/failures/{name}", combined)


# ---------------- SAVE METRICS ----------------
if has_gt:
    mean_iou = np.mean(all_ious)
else:
    mean_iou = -1

with open("outputs/metrics.txt", "w") as f:
    if has_gt:
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
    else:
        f.write("No ground truth available → IoU not computed\n")

print("🔥 Mean IoU:", mean_iou)


# ---------------- SAVE SUMMARY ----------------
with open("outputs/summary.txt", "w") as f:
    f.write("Semantic Segmentation Results\n")
    f.write("=============================\n\n")

    f.write(f"Total Images Tested: {len(dataset)}\n")

    if has_gt:
        f.write(f"Mean IoU: {mean_iou:.4f}\n\n")
    else:
        f.write("IoU: Not available (no ground truth)\n\n")

    f.write("Folders:\n")
    f.write("- comparisons/: Original | GT | Prediction | Overlay\n")
    f.write("- overlays/: Overlay visualization\n")
    f.write("- predictions/: Raw predicted masks\n")
    f.write("- failures/: Low IoU cases (if GT available)\n")