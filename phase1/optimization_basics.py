import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
convex = x ** 2
non_convex = x ** 4 - 4*x**2
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(x, convex)
ax[0].set_title("Convex: One minimum")
ax[1].plot(x, non_convex)
ax[1].set_title("Non-convex: Two minima")
plt.savefig('non_convex.png')
# plt.show()

def f(y):
    return y ** 4 - 4 * y ** 2

def f_prime(y):
    return 4 * y ** 3 - 8 * y

for z in np.linspace(-3, 3, 13):
    if abs(f_prime(z)) < 0.5:
        print(f"Near flat region around x = {z:.2f}, f'(x) = {f(z):.2f}")


#LEARNING RATE
def f(number):
    return number ** 2

def f_prime(number):
    return 2 * number

def run_gradient_descent(learning_rate, steps=15, start=5.0):
    point = start
    history = [point]
    for _ in range(steps):
        point = point - learning_rate * f_prime(point)
        history.append(point)
    return history

for lr in [0.01, 0.1, 0.9, 1.01]:
    print(f"Learning rate = {lr}")
    print([round(v, 2) for v in run_gradient_descent(lr)])
    plt.figure(figsize=(12, 5))
    plt.plot(run_gradient_descent(lr), label=f"lr={lr}")
    plt.title(f"Learning rate = {lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"loss_{lr}.png")