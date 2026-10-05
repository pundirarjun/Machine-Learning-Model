from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = PROJECT_ROOT / "training_history.npz"

OUTPUT_PATH = PROJECT_ROOT / "Training vs Validation RMSE plot.png"


history = np.load(HISTORY_PATH)

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

plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=150)

print(f"Saved plot to {OUTPUT_PATH}")

plt.show()
