from torchvision import datasets
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import pickle as pickle
import torch


def _to_vw_format(x,y, label=True):
    if y == 0 or y == -1:
        y = -1
    else:
        y = 1
    if label:
        s = '{} | '.format(y)
    else:
        s='| '
    s+='{}'.format(''.join([f'v{i}:{x[i]} ' for i in range(len(x))]))
    return s


def vw_format(X,y, name, label=True):
    with open(f'{name}.vw', 'w') as f:
        for x, y in tqdm(zip(X, y)):
            f.write(_to_vw_format(x,y, label=label)+'\n')


def get_cifar64_data():
    rnd = np.random.RandomState(1234)
    with open('../data/cifar_64.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    train_perm = rnd.permutation(X_data.shape[0])
    return X_data[train_perm], Y_data[train_perm], X_test_data, Y_test_data


def get_fashion_data():
    rnd = np.random.RandomState(1234)
    with open('../data/fashion_784.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    train_perm = rnd.permutation(X_data.shape[0])
    return X_data[train_perm], Y_data[train_perm], X_test_data, Y_test_data


def get_svhn_data():
    rnd = np.random.RandomState(1234)
    with open('../data/svhn_512.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    train_perm = rnd.permutation(X_data.shape[0])
    return X_data[train_perm], Y_data[train_perm], X_test_data, Y_test_data


def get_MNISThalf(d=784):
    rnd = np.random.RandomState(1234)
    raw_tr = datasets.MNIST('../data/MNIST', train=True, download=True)
    raw_te = datasets.MNIST('../data/MNIST', train=False, download=True)
    X_data = raw_tr.data.float().view(-1, 784) / 255.
    X_test_data = raw_te.data.float().view(-1, 784) / 255.
    X_mean = torch.mean(X_data, dim=0)
    X_data -= X_mean
    U, S, V = torch.svd(X_data)
    X_tr = torch.matmul(X_data, V[:, :d]).numpy()
    X_te = torch.matmul(X_test_data - X_mean, V[:, :d]).numpy()

    Y_tr = (raw_tr.targets > 4).float().numpy()
    Y_te = (raw_te.targets > 4).float().numpy()
    print('sizes', X_tr.size, Y_tr.size, X_te.size, Y_te.size)
    print('types', type(X_tr), type(Y_tr), type(X_te), type(Y_te))
    train_perm = rnd.permutation(50000)
    return X_tr[:50000][train_perm], Y_tr[:50000][train_perm], X_te, Y_te
