## Implementation of ACED

### File Structure
* `algorithm.py` Our main algorithmic implementation including grid search, stochastic mirror descent and waterfilling.
* `algorithm_gammastar.py` Result where we sample from gamma star, the optimal allocation when knowing h_star 
(optimal hypothesis).
* `passive.py` Implementation of the passive learning setting (query uniformly at random).
* `argmax_oracle.py` The weighted classification oracle where it returns the classification based on the model class 
and dataset.
* `dataset.py` Data loading script for different datasets.
* `hyperparam.py` The entry point of our implementation.
* `record.py` Class for logging our results.
* `utils.py` Utility file shared across directory.
* `preproc_cifar.py` Preprocessing CIFAR dataset.
* `preproc_fashion.py` Preprocessing FashionMNIST dataset.
* `preproc_svhn.py` Preprocessing SVHN dataset.
* `aggregate_accuracy.py` Plotting from log.


Then run the scripts in `src` with the proper command line arguments, e.g.,
```
rm -rf runs; rm log/*
python argmax_oracle.py mnist mlp exp_grad
```
The first command here clears out any log files! If you are trying to save runs, then don't run it.

To run tensorboard
```
    nohup tensorboard --logdir=runs --bind_all --reload_multifile True&
```
