import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 200)
y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Batch Gradient Descent
w, b = 0, 0
learning_rate = 0.01
batch_losses = []
for epoch in range(50):
    y_pred = w * X + b
    error = y_pred - y
    dw = (2 / len(X)) * np.sum(error * X)
    db = (2 / len(X)) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db
    batch_losses.append(mse(y, w * X + b))
print(f"Batch GD result: w={w:.3f}, b={b:.3f}")

# Stochastic Gradient Descent
w, b = 0, 0
learning_rate = 0.01
sgd_losses = []
for epoch in range(50):
    indices = np.random.permutation(len(X))
    for i in indices:
        xi, yi = X[i], y[i]
        y_pred = w * xi + b
        error = y_pred - yi
        dw = 2 * error * xi
        db = 2 * error

        w -= learning_rate * dw
        b -= learning_rate * db
    sgd_losses.append(mse(y, w * X + b))
print(f"Stochastic GD result: w={w:.3f}, b={b:.3f}")

# Plot loss curves after every epoch
epochs = np.arange(1, 51)
plt.figure(figsize=(10, 6))
plt.plot(epochs, batch_losses, label="Batch GD", color="tab:blue", linewidth=2)
plt.plot(epochs, sgd_losses, label="SGD", color="tab:orange", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Loss per epoch for Batch GD and SGD")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_image")
os.makedirs(RESULT_DIR, exist_ok=True)
output_path = os.path.join(RESULT_DIR, "gradient_descent_loss_curves.png")
plt.savefig(output_path)
print(f"Saved loss plot to {output_path}")