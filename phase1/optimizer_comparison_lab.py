import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), "result_image")
os.makedirs(RESULT_DIR, exist_ok=True)


def loss_surface(x, y):
    valley = x**2 + 5 * (y - x**2) ** 2
    wiggle = 0.05 * np.sin(5 * x)
    return valley + wiggle


def grad_loss(x, y):
    dx = 2 * x - 20 * x * (y - x**2) + 0.25 * np.cos(5 * x)
    dy = 10 * (y - x**2)
    return np.array([dx, dy])


def run_vanilla(start, lr=0.005, steps=200):
    theta = np.array(start, dtype=float).copy()
    path = [theta.copy()]
    losses = [loss_surface(*theta)]
    for _ in range(steps):
        g = grad_loss(*theta)
        theta = theta - lr * g
        path.append(theta.copy())
        losses.append(loss_surface(*theta))
    return np.array(path), np.array(losses)


def run_momentum(start, lr=0.015, beta=0.85, steps=200):
    theta = np.array(start, dtype=float).copy()
    velocity = np.zeros_like(theta)
    path = [theta.copy()]
    losses = [loss_surface(*theta)]
    for _ in range(steps):
        g = grad_loss(*theta)
        velocity = beta * velocity - lr * g
        theta = theta + velocity
        path.append(theta.copy())
        losses.append(loss_surface(*theta))
    return np.array(path), np.array(losses)


def run_adam(start, lr=0.02, beta1=0.9, beta2=0.99, steps=200):
    theta = np.array(start, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    path = [theta.copy()]
    losses = [loss_surface(*theta)]
    for t in range(1, steps + 1):
        g = grad_loss(*theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        path.append(theta.copy())
        losses.append(loss_surface(*theta))
    return np.array(path), np.array(losses)


def summarise(results):
    names = list(results.keys())
    best = {name: min(losses) for name, (_, losses) in results.items()}
    thresholds = {
        name: min(losses) + 0.05 * max(1.0, abs(min(losses)))
        for name, (_, losses) in results.items()
    }
    fastest = min(
        names,
        key=lambda name: next(i for i, loss in enumerate(results[name][1]) if loss <= thresholds[name]),
    )
    smoothness = {name: np.std(np.diff(losses)) for name, (_, losses) in results.items()}
    smoothest = min(names, key=lambda name: smoothness[name])
    oscillatory = [name for name, (_, losses) in results.items() if np.mean(np.diff(np.sign(np.diff(losses)))) > 0.05]

    summary_lines = [
        f"Fastest to reach near-minimum: {fastest}.",
        f"Smoothest descent curve: {smoothest}.",
        f"Oscillation/instability markers: {', '.join(oscillatory) if oscillatory else 'none detected'}.",
    ]
    return " ".join(summary_lines)


if __name__ == "__main__":
    start = np.array([-2.0, 3.0])
    steps = 200

    results = {
        "Vanilla GD": run_vanilla(start, lr=0.005, steps=steps),
        "Momentum": run_momentum(start, lr=0.015, beta=0.85, steps=steps),
        "Adam": run_adam(start, lr=0.02, beta1=0.9, beta2=0.99, steps=steps),
    }

    xs = np.linspace(-4, 4, 400)
    ys = np.linspace(-1.5, 4.0, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = loss_surface(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contour(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")
    for name, (path, _) in results.items():
        color = {"Vanilla GD": "tab:red", "Momentum": "tab:blue", "Adam": "tab:green"}[name]
        ax.plot(path[:, 0], path[:, 1], marker='o', markersize=2.5, linewidth=1.5, color=color, label=name)
    ax.scatter(start[0], start[1], color="black", s=40, marker="x", label="Start")
    ax.set_title("Optimizer paths on a toy non-convex surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "optimizer_paths.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (_, losses) in results.items():
        color = {"Vanilla GD": "tab:red", "Momentum": "tab:blue", "Adam": "tab:green"}[name]
        ax.plot(np.arange(len(losses)), losses, label=name, color=color, linewidth=2)
    ax.set_title("Loss vs training step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "optimizer_loss_curves.png"))

    print("Final losses:")
    for name, (_, losses) in results.items():
        print(f"  {name}: {losses[-1]:.6f}")
    print()
    print(summarise(results))
