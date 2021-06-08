from src.hyperparam import *
import torchvision.datasets as datasets
import pickle
import multiprocessing as mp
import torch
import numpy as np


data_dict = {}


def init_worker(X, X_shape, Y, Y_shape, X_test, X_test_shape, Y_test, Y_test_shape):
    # X and Y should be multiprocessing.RawArray with type float
    data_dict['X'] = X
    data_dict['X_shape'] = X_shape
    data_dict['Y'] = Y
    data_dict['Y_shape'] = Y_shape
    data_dict['X_test'] = X_test
    data_dict['X_test_shape'] = X_test_shape
    data_dict['Y_test'] = Y_test
    data_dict['Y_test_shape'] = Y_test_shape


# Load data from RawArray shared among different processes.
def get_dataset():
    dataset = {"X": torch.from_numpy(np.frombuffer(data_dict["X"], dtype=np.dtype(data_dict["X"]))).reshape(
        data_dict["X_shape"]),
        "Y": torch.from_numpy(np.frombuffer(data_dict["Y"], dtype=np.dtype(data_dict["Y"]))).reshape(
            data_dict["Y_shape"]),
        "X_test": torch.from_numpy(np.frombuffer(data_dict["X_test"], dtype=np.dtype(data_dict["X_test"]))).reshape(
            data_dict["X_test_shape"]),
        "Y_test": torch.from_numpy(np.frombuffer(data_dict["Y_test"], dtype=np.dtype(data_dict["Y_test"]))).reshape(
            data_dict["Y_test_shape"])
    }
    return dataset


def get_batch(idx, dataset, y=None):
    '''
    idx (int or list of ints): Indexes through the dataset to retrieve a minibatch.
    '''
    if y is not None:
        return dataset["X"][idx], y[idx].float()
    else:
        return dataset["X"][idx], dataset["Y"][idx]


def get_shared_data():
    '''
    Create shared data object based on the data_name specified at command line. data_name is loaded from hyperparam.py.
    '''
    if data_name == "mnist_half":
        return get_mnist_half_data(model_name not in ["cnn", "lenet"])
    elif data_name == "cifar64":
        return get_cifar64_data()
    elif data_name == "fashion":
        return get_fashion_data()
    elif data_name == "svhn":
        return get_svhn_data()
    elif data_name == "2d_test":
        return get_2d_test_data()
    elif data_name == "2d_sample":
        return get_2d_sample_data()


def build_raw_array(data):
    X = mp.RawArray('f', data.size)
    X_np = np.frombuffer(X, dtype=np.dtype(X)).reshape(data.shape)
    np.copyto(X_np, data)
    return X


def mnist_data_helper(flatten, train):
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=None)
    X_data = dataset.data.float() / 255.
    if flatten:
        X_data = X_data.view(-1, 28 * 28)
    else:
        X_data = X_data.unsqueeze(1)  # color channel, so once loaded the images are batch_size * 1 * width * height
    Y_data = dataset.targets
    return X_data, Y_data


def get_mnist_half_data(flatten, pca_dim=PCA_dim):
    X_data, Y_data = mnist_data_helper(flatten, True)
    X_test_data, Y_test_data = mnist_data_helper(flatten, False)
    X_mean = torch.mean(X_data, dim=0)
    X_data -= X_mean
    U, S, V = torch.svd(X_data)
    X_data = torch.matmul(X_data, V[:, :pca_dim])
    X_test_data = torch.matmul(X_test_data - X_mean, V[:, :pca_dim]).numpy()

    Y_data = (Y_data > 4).float()
    Y_test_data = (Y_test_data > 4).float().numpy()

    X_data = X_data.numpy()[:N_dim]
    Y_data = Y_data.numpy()[:N_dim]
    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_cifar128_data():
    with open('../data/cifar_128.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_cifar64_data():
    with open('../data/cifar_64.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_fashion_data():
    with open('../data/fashion_784.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_svhn_data():
    with open('../data/svhn_512.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_2d_test_data():
    rnd = np.random.RandomState(12345)
    X_data = rnd.rand(N_dim, 2)
    Y_data = (((X_data[:, 0] > .5).astype(int) + (X_data[:, 1] > .5).astype(int)) > 0).astype(int)
    X_test_data = rnd.rand(N_dim, 2)
    Y_test_data = (((X_test_data[:, 0] > .5).astype(int) + (X_test_data[:, 1] > .5).astype(int)) > 0).astype(int)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


def get_2d_sample_data():
    with open('../data/2d_sample_3.pkl', 'rb') as file:
        X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

    return build_raw_array(X_data), X_data.shape, build_raw_array(Y_data), Y_data.shape, \
           build_raw_array(X_test_data), X_test_data.shape, build_raw_array(Y_test_data), Y_test_data.shape


if __name__ == "__main__":
    print(get_svhn_data())
