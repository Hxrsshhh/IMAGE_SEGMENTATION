import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, img_size=256):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        # Ensure sorted order so image and mask match
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # -------- SAFETY CHECK --------
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_name}")

        # -------- READ IMAGE --------
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to load image {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, (self.img_size, self.img_size))

        # -------- READ MASK --------
        mask = cv2.imread(mask_path, 0)

        if mask is None:
            raise ValueError(f"Failed to load mask {mask_path}")

        # Resize mask with NEAREST interpolation
        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -------- AUGMENTATION --------
        if self.transform:

            augmented = self.transform(image=image, mask=mask)

            image = augmented["image"]
            mask = augmented["mask"]

        # -------- NORMALIZATION --------
        image = image.astype(np.float32) / 255.0

        # Better normalization for pretrained encoders
        image = (image - 0.5) / 0.5

        # -------- TO TENSOR --------
        image = torch.from_numpy(image).permute(2, 0, 1).float().contiguous()
        mask = torch.from_numpy(mask).long()

        return image, mask