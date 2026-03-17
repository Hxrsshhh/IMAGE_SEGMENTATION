import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


# -------------------------------
# CLASS MAPPING
# -------------------------------
CLASS_MAP = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9   # Sky
}

NUM_CLASSES = len(CLASS_MAP)

# Ignore index for unknown pixels
IGNORE_INDEX = 255


# -------------------------------
# OPTIONAL: CLASS NAMES (for report)
# -------------------------------
CLASS_NAMES = {
    0: "Trees",
    1: "Lush Bushes",
    2: "Dry Grass",
    3: "Dry Bushes",
    4: "Ground Clutter",
    5: "Flowers",
    6: "Logs",
    7: "Rocks",
    8: "Landscape",
    9: "Sky"
}


# -------------------------------
# ENCODE MASK FUNCTION
# -------------------------------
def encode_mask(mask):
    """
    Convert original mask values → class indices
    Unknown values → IGNORE_INDEX
    """
    encoded = np.ones_like(mask, dtype=np.uint8) * IGNORE_INDEX

    for original_id, new_id in CLASS_MAP.items():
        encoded[mask == original_id] = new_id

    return encoded


# -------------------------------
# DATASET CLASS
# -------------------------------
class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, img_size=256, debug=False):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.debug = debug

        # Only valid image files
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        # -------------------------------
        # IMAGE PATH
        # -------------------------------
        img_path = os.path.join(self.image_dir, img_name)

        # -------------------------------
        # HANDLE EXTENSION MISMATCH
        # -------------------------------
        mask_name = img_name
        if img_name.endswith(".jpg"):
            mask_name = img_name.replace(".jpg", ".png")
        elif img_name.endswith(".jpeg"):
            mask_name = img_name.replace(".jpeg", ".png")

        mask_path = os.path.join(self.mask_dir, mask_name)

        # -------------------------------
        # SAFETY CHECKS
        # -------------------------------
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for: {img_name}")

        # -------------------------------
        # READ IMAGE
        # -------------------------------
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # -------------------------------
        # READ MASK
        # -------------------------------
        mask = cv2.imread(mask_path, 0)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------
        # ENCODE MASK (CRITICAL)
        # -------------------------------
        mask = mask.astype(np.uint8)

        # -------------------------------
        # AUGMENTATION
        # -------------------------------
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # -------------------------------
        # NORMALIZATION
        # -------------------------------
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        # -------------------------------
        # TO TENSOR
        # -------------------------------
        image = torch.from_numpy(image).permute(2, 0, 1).float().contiguous()
        mask = torch.from_numpy(mask).long()

        # -------------------------------
        # DEBUG MODE
        # -------------------------------
        if self.debug and index == 0:
             print("✅ DEBUG INFO")
             print("Image shape:", image.shape)
             print("Mask shape:", mask.shape)

             print("RAW MASK VALUES:", np.unique(mask))  # ADD THIS

             print("Mask unique values:", torch.unique(mask))

        if IGNORE_INDEX in torch.unique(mask):
                print("⚠️ Found IGNORE pixels (255)")

        # -------------------------------
        # RETURN
        # -------------------------------
        return image, mask, img_name