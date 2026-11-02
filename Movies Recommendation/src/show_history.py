import numpy as np

h = np.load("training_history.npz")

print("keys", list(h.files))

tr, va, lr = h["train_rmse"], h["val_rmse"], h["lr"]

print("epochs", len(va))

for e in [0, 4, 9, 14, 19, 29, 39, 49, 54, 59]:
    print(f"ep{e + 1:3d}  train {tr[e]:.4f}  val {va[e]:.4f}  lr {lr[e]:.5f}")

print("best val", round(float(va.min()), 4), "at epoch", int(va.argmin()) + 1)
print("final train", round(float(tr[-1]), 4), "final lr", round(float(lr[-1]), 5))
