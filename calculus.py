import numpy as np

def compute_square(value):
    return value**2

def numerical_derivative(func, x, h=1e-5):
    return (func(x+h) - func(x)) / h

for i in [1, 3, -2]:
    print(f"f'({i}) = {numerical_derivative(compute_square, i):.4f} (true value: {2*i})")


def function_example1(x, y):
    return (x ** 2) * y + 3 * y

def partial_derivative_x(func, x, y, h=1e-5):
    return (func(x+h, y) - func(x, y)) / h

def partial_derivative_y(func, x, y, h=1e-5):
    return (func(x, y+h) - func(x, y)) / h

print(f"Partial derivative w.r.t x: ∂f/∂x at (2, 1) = {partial_derivative_x(function_example1, 2, 1)}")
print(f"Partial derivative w.r.t y: ∂f/∂y at (2, 1) = {partial_derivative_y(function_example1, 2, 1)}")

def function_example2(x, y):
    return x ** 2 + y ** 2

def gradient(x, y):
    return np.array([2*x, 2*y])

point = np.array([3, 4])
learning_rate = 0.1

for step in range(20):
    g = gradient(point[0], point[1])
    point = point - learning_rate * g
    print(f"step: {step}, g: {g}, point: {point}, f={function_example2(point[0], point[1]):.4f}")