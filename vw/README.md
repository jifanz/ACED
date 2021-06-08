# Vowpal Wabbit Experiments

##Requirement:
Vowpal Wabbit version 8.9.0.

##Beforehand
Run `src/preproc_cifar.py`, `src/preproc_fashion.py`, `src/preproc_svhn.py` before running this.

##Reproducing Our Results:
```
mkdir log
python stacked.py --data svhn --batch 250 --passes 2 --c0 1e-2 1e-1 1 10 100
python stacked.py --data fashion --batch 250 --passes 2 --c0 1e-2 1e-1 1 10 100
python stacked.py --data cifar64 --batch 250 --passes 2 --c0 1e-2 1e-1 1 10 100
python stacked.py --data mnist_half --batch 250 --passes 1 --c0 1e-2 1e-1 1 10 100
python stacked.py --data mnist_half --batch 250 --passes 1 --c0 1e-2 1e-1 1 10 100 --ora
```