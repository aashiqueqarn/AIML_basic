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