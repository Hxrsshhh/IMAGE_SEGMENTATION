import torch
import cv2
import numpy as np

from models.model import get_model

MODEL_PATH = "best_model.pth"
NUM_CLASSES = 40
IMG_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def predict(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0
    image = (image - 0.5) / 0.5

    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(image)

    mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    return mask


def color_mask(mask):

    colors = np.random.randint(0,255,(NUM_CLASSES,3))

    h,w = mask.shape
    color_mask = np.zeros((h,w,3),dtype=np.uint8)

    for c in range(NUM_CLASSES):
        color_mask[mask==c] = colors[c]

    return color_mask


mask = predict("test.jpg")

colored = color_mask(mask)

cv2.imwrite("prediction.png", colored)

print("Prediction saved as prediction.png")