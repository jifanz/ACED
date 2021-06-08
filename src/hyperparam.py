import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

all_data = ["mnist_half", "cifar64", "svhn", "fashion", "2d_test", "2d_sample"]
all_model = ["sklogistic", "svm", "sklinear", "2d_thresh"]

assert len(sys.argv) >= 3, "Usage: python <python file> <dataset name> <model class>"
data_name = sys.argv[1]
model_name = sys.argv[2]
assert data_name in all_data
assert model_name in all_model

'''
N_dim: number of data points in dataset.
N_max_processes: pool size for multi-processing.
N_it_gamma_combi: number of iterations for optimizing lambda when computing allocation.
'''
PCA_dim = None
if data_name == "mnist_half":
    N_dim = 50000
    N_epoch_argmax = 100
    N_max_processes = 50
    N_it_gamma_combi = 100
    PCA_dim = 784
elif data_name == "cifar64":
    N_dim = 10000
    N_epoch_argmax = 100
    N_max_processes = 50
    N_it_gamma_combi = 100
elif data_name == "fashion":
    N_dim = 12000
    N_epoch_argmax = 30
    N_max_processes = 50
    N_it_gamma_combi = 100
elif data_name == "svhn":
    N_dim = 16180
    N_epoch_argmax = 100
    N_max_processes = 50
    N_it_gamma_combi = 100
elif data_name == "2d_test":
    N_dim = 10000
    N_max_processes = 50
    N_it_gamma_combi = 100
elif data_name == "2d_sample":
    N_dim = 2000
    N_max_processes = 50
    N_it_gamma_combi = 100
