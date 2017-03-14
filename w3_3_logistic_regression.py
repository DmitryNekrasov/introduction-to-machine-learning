from math import exp
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas


def sigmoid(weights, x, y):
    return 1.0 / (1.0 + exp(y * np.dot(weights, x)))


def gradient(x, y, k=0.1, c=0, eps=1e-5, max_iter=100000):
    l = len(x)
    w = np.array((0, 0))
    for it in range(max_iter):
        s0 = s1 = 0
        for i in range(l):
            sig = sigmoid(w, np.array((x[1][i], x[2][i])), -y[i])
            s0 += y[i] * x[1][i] * (1.0 - sig)
            s1 += y[i] * x[2][i] * (1.0 - sig)
        w0 = w[0] + k / l * s0 - k * c * w[0]
        w1 = w[1] + k / l * s1 - k * c * w[1]
        new_w = np.array((w0, w1))
        error = np.linalg.norm(w - new_w)
        if error < eps:
            break
        w = new_w
    print('it =', it)
    return w


def get_auc_roc(x, y, w):
    a = [sigmoid(-w, vec_x, 1) for vec_x in np.array(x)]
    val = roc_auc_score(y, a)
    return round(val, 3)


data = pandas.read_csv('samples/data-logistic.csv', header=None)
target = data[0]
attributes = data.drop(0, axis=1)

w_c0 = gradient(attributes, target)
w_c10 = gradient(attributes, target, c=10)

auc_roc_c0 = get_auc_roc(attributes, target, w_c0)
auc_roc_c10 = get_auc_roc(attributes, target, w_c10)

print(auc_roc_c0, auc_roc_c10)
