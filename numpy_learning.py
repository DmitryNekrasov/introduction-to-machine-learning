import numpy as np

X = np.random.normal(loc=1, scale=10, size=(5, 5))
print(X)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std
print(X_norm)

Z = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, -8, 9],
              [10, 11, 12],
              [13, 14, -15]])
print(Z)

sum_row = np.sum(Z, axis=1)
print(sum_row)
print(np.nonzero(sum_row > 10))

A = np.eye(3)
B = np.eye(3)
AB = np.vstack((A, B))
print(A)
print(B)
print(AB)
