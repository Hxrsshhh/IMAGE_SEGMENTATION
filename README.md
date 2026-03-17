# 🌄 Offroad Semantic Segmentation (Image Segmentation Project)

A deep learning project for **semantic segmentation of off-road environments**, trained on synthetic desert data.
The model predicts **pixel-wise class labels** to identify terrain, vegetation, obstacles, and sky.

---

## 🚀 Features

* ✅ Pixel-level semantic segmentation
* ✅ Trained on synthetic desert dataset
* ✅ Evaluation using **IoU (Intersection over Union)**
* ✅ Visual outputs (Original | GT | Prediction | Overlay)
* ✅ Failure case analysis
* ✅ Clean training + testing pipeline

---

## 📂 Dataset

Download dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/hxrsshhh/test-images

---

### 📁 Folder Structure

Place dataset like this:

```
dataset/
│
├── train/
│   ├── images/
│   ├── masks/
│
├── val/
│   ├── images/
│   ├── masks/
│
├── test/
│   ├── images/
│   ├── masks/   # optional
```

---

## 🧠 Model

* Architecture: Custom segmentation model (UNet/DeepLab-like)
* Input size: 256×256
* Classes: 40
* Loss: CrossEntropyLoss
* Metric: IoU (Intersection over Union)

---

## 📥 Download Trained Model

Due to GitHub size limits, model is hosted externally:

🔗 **Download Model Weights:**
https://your-google-drive-link-here

Place it in project root:

```
best_model.pth
```

---

## ⚙️ Installation

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

---

## 🏋️ Training

```bash
python train.py
```

### Output:

* `best_model.pth`
* `checkpoint.pth`
* `train_losses.npy`
* `val_losses.npy`

---

## 🧪 Testing & Inference

```bash
python test.py
```

---

## 📊 Outputs

Generated in:

```
outputs/
│
├── comparisons/   # Original | GT | Prediction | Overlay
├── overlays/      # Overlay only
├── predictions/   # Raw segmentation
├── failures/      # Low IoU cases
│
├── metrics.txt    # IoU score
├── summary.txt
```

---

## 📈 Loss Graph

```bash
python plot_loss.py
```

Generates:

```
loss.png
```

---

## 📊 Results

| Metric   | Value                                     |
| -------- | ----------------------------------------- |
| Mean IoU | ~0.55–0.65 (expected after full training) |

---

## 🧠 Example Output

Each prediction shows:

```
| Original | Ground Truth | Prediction | Overlay |
```

---

## ⚠️ Challenges Faced

* Handling multi-class segmentation (40 classes)
* Dataset label understanding (no mapping needed)
* Model generalization on unseen terrain

---

## 🔍 Failure Cases

* Small objects misclassified
* Similar textures (grass vs bushes)
* Low contrast regions

---

## 🚀 Future Improvements

* Use DeepLabV3+ / UNet++
* Add data augmentation
* Train on GPU for faster convergence
* Reduce classes (40 → 10 semantic groups)

---

## 📌 Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib

---

## 👨‍💻 Author

**Happy**
BTech Student | ML Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
