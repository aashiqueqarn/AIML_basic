import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 50)
y_true = 2 * X + 1
y = y_true - np.random.normal(0, 1, size=X.shape)
w, b = 0, 0
learning_rate = 0.00001
loss_history = []

for epoch in range(1000):
    y_pred = w * X + b
    error = y_pred - y
    dw = (2 / len(X)) * np.sum(error * X)
    db = (2 / len(X)) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: w={w:.3f}, b={b:.3f} | Loss: {loss:.3f}")

print(f"Final w={w:.3f}, b={b:.3f} (true values were w=2, b=1)")
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(loss_history, color='blue', linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Loss Curve: Training process")
plt.grid(True, alpha=0.3)
plt.yscale('log')
final_loss = loss_history[-1]
plt.axhline(y=final_loss, color='red', linestyle='--', alpha=0.5, label=f"Final Loss = {final_loss:.3f}")
plt.legend()


plt.subplot(1,2,2)
plt.scatter(X, y_true, color='blue', alpha=0.6, label='Noisy Data')
plt.plot(X, y_true, 'g-', label='True: y = 2x + 1', linewidth=2)
plt.plot(X, w * X + b, 'b-', label=f'Learned: y = {w:.3f}x + {b:.3f}', linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Final Model Fit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gradient_descent_noisy.png")
plt.show()