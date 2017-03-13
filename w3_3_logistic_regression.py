from math import exp
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas


def sigmoid(weights, x, y):
    return 1.0 / (1.0 + exp(y * np.dot(weights, x)))


data = pandas.read_csv('samples/data-logistic.csv', header=None)
target = data[0]
attributes = data.drop(0, axis=1)

l = len(attributes)
w = np.array((0, 0))
k = 0.1
s0 = 0
s1 = 0
C = 0
eps = 1e-5
max_iter = 10000

for it in range(max_iter):
    s0 = s1 = 0
    for i in range(l):
        sig = sigmoid(w, np.array((attributes[1][i], attributes[2][i])), -target[i])
        s0 += target[i] * attributes[1][i] * (1.0 - sig)
        s1 += target[i] * attributes[2][i] * (1.0 - sig)
    w0 = w[0] + k / l * s0 - k * C * w[0]
    w1 = w[1] + k / l * s1 - k * C * w[1]
    new_w = np.array((w0, w1))
    error = np.linalg.norm(w - new_w)
    if error < eps:
        break
    w = new_w
    print(w)
    k += 0.1

print('it =', it)

a = [sigmoid(-w, x, 1) for x in np.array(attributes)]
val = roc_auc_score(target, a)
print(val)
