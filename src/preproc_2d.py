import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d
from numpy import sin, cos, pi, exp

np.random.seed(123)

C = 3


# SAMPLING ROUTINE
def p(a, b, k=2):
    c = sin(pi * k) ** 3 * cos(pi * k) / (pi ** 2 * k ** 2) + 1
    return np.abs(a - b) * (sin(2 * pi * k * a) * cos(2 * pi * k * b) + 1)


def f(z, c=3):
    return np.random.binomial(1, p=(1 + exp(c * (z[1] - 1 + z[0]))) ** (-1))


def sampler(n_points, p=None):
    accepted = []
    while len(accepted) < n_points:
        x, y, z = np.random.rand(3)
        if z < p(x, y):
            accepted.append([x, y])
    return np.array(accepted)


def get_labels(X, f=None):
    y = []
    for x in X:
        y.append(f(x))
    return y


X = np.array(sampler(2000, p=p))
y = np.array(get_labels(X, f=f))
X_test = np.array(sampler(500, p=p))
y_test = np.array(get_labels(X_test, f=f))
plt.scatter(X[:, 0], X[:, 1], c=['red' if l == 1 else 'blue' for l in y], s=3)
plt.show()

with open("../data/2d_sample_{}.pkl".format(C), 'wb') as file:
    pickle.dump([X.astype(float), y.astype(float), X_test.astype(float),
                 y_test.astype(float)], file, protocol=pickle.HIGHEST_PROTOCOL)
