import numpy as np
import matplotlib.pyplot as plt

train_losses = np.load("train_losses.npy")
val_losses = np.load("val_losses.npy")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()

plt.savefig("loss.png")
plt.show()