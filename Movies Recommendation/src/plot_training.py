import numpy as np
import matplotlib.pyplot as plt

history = np.load(r"D:\Machine Learning Model\Movies Recommendation\training_history.npz")

train_rmse = history["train_rmse"]
val_rmse = history["val_rmse"]

epochs = np.arange(1, len(train_rmse) + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_rmse,
    marker="o",
    label="Training RMSE"
)

plt.plot(
    epochs,
    val_rmse,
    marker="o",
    label="Validation RMSE"
)

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE")
plt.legend()
plt.grid(True)

plt.show()