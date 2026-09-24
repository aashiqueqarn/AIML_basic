import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_image")
os.makedirs(RESULT_DIR, exist_ok=True)


def f(x, y):
    return 0.1 * x**2 + 5 * y**2

def grad_f(x, y):
    return np.array([0.2 * x, 10 * y])

# Vanilla Gradient Descent
point = np.array([-8.0, 3.0])
learning_rate = 0.02
path_vanilla = [point.copy()]
for _ in range(100):
    point = point - learning_rate * grad_f(point[0], point[1])
    path_vanilla.append(point.copy())

# Momentum Gradient Descent
point = np.array([-8.0, 3.0])
velocity = np.array([0.0, 0.0])
beta = 0.5
path_momentum = [point.copy()]
for _ in range(100):
    g = grad_f(point[0], point[1])
    velocity = beta * velocity - learning_rate * g
    point = point + velocity
    path_momentum.append(point.copy())

# Convert paths to NumPy arrays
path_vanilla = np.array(path_vanilla)
path_momentum = np.array(path_momentum)

print("Vanilla Gradient Descent Path:", path_vanilla[-1])
print("Momentum Gradient Descent Path:", path_momentum[-1])

# Plot trajectories with contour of f underneath
xs = np.linspace(-9, 1, 200)
ys = np.linspace(-1, 5, 200)
Xg, Yg = np.meshgrid(xs, ys)
Z = f(Xg, Yg)

plt.figure(figsize=(8, 6))
plt.contour(Xg, Yg, Z, levels=30, cmap="viridis", alpha=0.6)
plt.plot(path_vanilla[:, 0], path_vanilla[:, 1], '-o', color='red', label='Vanilla GD', markersize=3)
plt.plot(path_momentum[:, 0], path_momentum[:, 1], '-o', color='blue', label='Momentum GD', markersize=3)
plt.scatter([path_vanilla[0,0]], [path_vanilla[0,1]], color='black', marker='x', label='Start')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectories: Vanilla vs Momentum Gradient Descent')
plt.legend()
plt.grid(alpha=0.3)
# Lock axis limits to the contour grid so tiny/huge values from divergence remain visible
plt.xlim(xs.min(), xs.max())
plt.ylim(ys.min(), ys.max())
plt.tight_layout()
output_path = os.path.join(RESULT_DIR, 'optimization_momentum_paths.png')
plt.savefig(output_path)
print(f"Saved trajectory plot to {output_path}")

# Zoomed-in view around the valley to better compare trajectories
plt.figure(figsize=(8, 6))
plt.contour(Xg, Yg, Z, levels=30, cmap="viridis", alpha=0.6)
plt.plot(path_vanilla[:, 0], path_vanilla[:, 1], '-o', color='red', label='Vanilla GD', markersize=4)
plt.plot(path_momentum[:, 0], path_momentum[:, 1], '-o', color='blue', label='Momentum GD', markersize=4)
plt.scatter([path_vanilla[0,0]], [path_vanilla[0,1]], color='black', marker='x', label='Start')
plt.xlim(-9, -1)
plt.ylim(-0.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Zoomed: Trajectories near the valley')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
output_path_zoom = os.path.join(RESULT_DIR, 'optimization_momentum_paths_zoom.png')
plt.savefig(output_path_zoom)
print(f"Saved zoomed trajectory plot to {output_path_zoom}")
