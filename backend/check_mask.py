import cv2
import numpy as np

mask = cv2.imread("dataset/train/masks/cc0000012.png", 0)

print("Unique mask values:", np.unique(mask))