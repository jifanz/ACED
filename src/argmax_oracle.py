"""
This file contains the files for computing the argmax oracle.

argmax_oracle is the general oracle to be called. We also implement individual argmax_oracle_<model> methods that are
specific to each type of model class.
"""
from sklearn import svm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from src.dataset import *
import numpy as np
import torch


def argmax_oracle_single_helper(name, w, seed=12345, visualize=False, return_test_accuracy=True):
    """
    w: a numpy array of weights that we take inner product with.
    """

    torch.random.manual_seed(seed)

    # Normalize and transform to weighted classification where each weight is in [0, 1].
    w = torch.from_numpy(w).float()
    w = w / torch.max(torch.abs(w))
    label = (w > 0).float()
    dataset = get_dataset()

    # Depending on the function class, we call different argmax oracles.
    w.abs_()
    if name == "sklogistic":
        return argmax_oracle_sklogistic(dataset, w, label, return_test_accuracy)
    elif name == "svm":
        return argmax_oracle_svm(dataset, w, label, return_test_accuracy)
    elif name == "sklinear":
        return argmax_oracle_sklinear(dataset, w, label, return_test_accuracy)
    elif name == "2d_thresh":
        return argmax_oracle_2d_thresh(dataset, w, label, return_test_accuracy)


def argmax_oracle_single(w, seed=12345, visualize=False, return_test_accuracy=True):
    return argmax_oracle_single_helper(model_name, w, seed=seed, visualize=visualize,
                                       return_test_accuracy=return_test_accuracy)


def argmax_oracle_2d_thresh(dataset, w, label, return_test_accuracy):
    # assuming only 0/1 in w
    rnd = np.random.RandomState(12345)
    hs = rnd.rand(10000, 2)

    X = dataset["X"].numpy()
    y = label.numpy()
    X_test = dataset["X_test"].numpy()
    y_test = dataset["Y_test"].numpy()
    X_train = X[w != 0]
    y_train = y[w != 0]
    thresh = (hs[:, 0].reshape((-1, 1)) > X_train[:, 0]).astype(int) + (
                hs[:, 1].reshape((-1, 1)) > X_train[:, 1]).astype(int)
    thresh = (thresh == 2).astype(int)
    best_thresh = np.argmax(np.sum((thresh == y_train.astype(int)).astype(int), axis=1))

    # print(hs[best_thresh])
    # import matplotlib.pyplot as plt
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=['red' if l == 1 else 'blue' for l in thresh[best_thresh]])
    # plt.show()

    pred = (hs[:, 0][best_thresh] > X[:, 0]).astype(int) + (hs[:, 1][best_thresh] > X[:, 1]).astype(int)
    pred_test = (hs[:, 0][best_thresh] > X_test[:, 0]).astype(int) + (
                hs[:, 1][best_thresh] > X_test[:, 1]).astype(int)

    if return_test_accuracy:
        return (pred == 2).astype(int), np.mean(((pred_test == 2).astype(int) == y_test).astype(float))
    else:
        return (pred == 2).astype(int)


def argmax_oracle_sklearn(model, dataset, w, label, return_test_accuracy):
    X = dataset["X"].numpy()
    y = label.numpy()
    X_test = dataset["X_test"].numpy()
    y_test = dataset["Y_test"].numpy()

    if np.sum(y[w != 0]) == 0:
        # All labels are 0.
        if return_test_accuracy:
            return np.zeros(y.shape), np.mean((y_test == 0).astype(float))
        else:
            return np.zeros(y.shape)
    elif np.sum(y[w != 0]) == len(y[w != 0]):
        # All labels are 1.
        if return_test_accuracy:
            return np.ones(y.shape), np.mean((y_test == 1).astype(float))
        else:
            return np.ones(y.shape)
    else:
        # Mixed labels, train.
        if X.shape[-1] == 1:
            # sklearn doesn't accept single dimension features, so pad with a dummy dimension.
            X = np.concatenate([X, np.ones(X.shape)], axis=-1)
            X_test = np.concatenate([X_test, np.ones(X_test.shape)], axis=-1)
        w = w.numpy()
        model.fit(X[w != 0], y[w != 0], w[w != 0])
        if return_test_accuracy:
            return model.predict(X), np.mean((model.predict(X_test) == y_test).astype(float))
        else:
            return model.predict(X)


def argmax_oracle_svm(dataset, w, label, return_test_accuracy):
    model = svm.SVC(C=1e10, kernel='poly', degree=2, max_iter=N_epoch_argmax, tol=1e-40)
    return argmax_oracle_sklearn(model, dataset, w, label, return_test_accuracy)


def argmax_oracle_sklogistic(dataset, w, label, return_test_accuracy):
    model = LogisticRegression(penalty="none", max_iter=N_epoch_argmax * (10 if return_test_accuracy else 1),
                               tol=.1 if data_name == "mnist_half" else .001, verbose=0, C=1e8, solver='lbfgs')
    return argmax_oracle_sklearn(model, dataset, w, label, return_test_accuracy)


def argmax_oracle_sklinear(dataset, w, label, return_test_accuracy):
    model = RidgeClassifier(alpha=1e-40, max_iter=N_epoch_argmax, tol=1e-40)
    return argmax_oracle_sklearn(model, dataset, w, label, return_test_accuracy)


if __name__ == "__main__":
    np.random.seed(12345)
    X, X_shape, Y, Y_shape, X_test, X_test_shape, Y_test, Y_test_shape = get_shared_data()
    init_worker(X, X_shape, Y, Y_shape, X_test, X_test_shape, Y_test, Y_test_shape)

    w = get_dataset()["Y"].numpy() * 2 - 1
    print("Best among every function", w @ (w > 0))
    import time

    t = time.time()
    for _ in range(1):
        z_hat, test_acc = argmax_oracle_single(w, visualize=True, return_test_accuracy=True)
    print("Best in function class:", w @ z_hat)

    print("Accuracy: ", np.sum((w > 0) == z_hat) / float(N_dim))
    print("Test Accuracy: ", test_acc)
    print(time.time() - t)
