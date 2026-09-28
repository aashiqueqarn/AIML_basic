import numpy as np
import time

array = np.array([10, 20, 30])
print(array + 5)

n = 5_000_000
data = list(range(n))

# Python loop version
start = time.time()
result_loop = [x * 2 for x in data]
loop_time = time.time() - start

# NumPy vectorized version
arr = np.array(data)
start = time.time()
result_vectorized = arr * 2
vectorized_time = time.time() - start

print(f"Python loop time: {loop_time:.6f} seconds")
print(f"NumPy vectorized time: {vectorized_time:.6f} seconds")
print(f"Speedup: {loop_time / vectorized_time:.2f}x faster with NumPy")

prices = np.array([100, 250, 80, 999, 45])
print(prices * 1.18)  # Adding 18% tax to each price
print(prices > 200)
print(prices[prices > 200])