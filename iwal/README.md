# Batched IWAL Experiments

##Beforehand
Run `src/preproc_cifar.py`, `src/preproc_fashion.py`, `src/preproc_svhn.py` before running this.

##Reproducing Our Results:
```
python run_iwal.py --data [data] --initqueries 250 --batch 250 --passes [passes] --thresholdtype [thresholdtype] --querystyle [querystyle] --clist [clist]
```
*`[data]` is one of `cifar64`, `svhn`, `fashion`, `mnist_half` denoting the dataset the experiment uses.

*`[passes]` is 1 for `mnist_half` and 2 for every other dataset, denoting the number of passes we feed the dataset to the IWAL algorithms.

*`[thresholdtype]` is one of `iwal0` or `iwal1` denoting the algorithm to run.

*`[querystyle]` is one of `reg` or `ora` denoting whether to use the variant of the algorithm.

*`[clist]` is a list of floating point numbers indicating the range of C_0 to search from (use the range in Appendix M for reproduction).

##Example Command:
```
mkdir log
python run_iwal.py --data=cifar64 --initqueries=250 --batch=250 --passes=2 --thresholdtype=iwal0 --querystyle=reg --clist 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1
```