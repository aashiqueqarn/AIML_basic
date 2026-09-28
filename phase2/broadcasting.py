import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([1, 0, -1])

print(A + b)
print(A + 100)

scores = np.array([
    [80, 90, 70],
    [60, 85, 95],
    [75, 70, 60]
])
col_mean = scores.mean(axis=0)
col_std = scores.std(axis=0)
normalized_scores = (scores - col_mean) / col_std
print(normalized_scores)