# Improved Algorithms for Agnostic Pool-based Active Classification

### File Structure
* `iwal/`   Our implementation of IWAL based methods, see `iwal/README.md` for details.
* `src/`    Our implementation of ACED, the algorithm proposed. See `src/README.md` for details.
* `vw/`     Our script for running Vowpal Wabbit implementations of Online Active Cover and oracular IWAL. 
See `vw/README.md` for details.
* `fig/`     Plots generated and included in the paper.
* `data/`    Directory where datasets resides.

### Python Environment Requirement
* `python`                    3.8.3
* `scikit-learn`              0.23.1
* `pytorch`                   1.7.0               [cpuonly]
* `torchvision`               0.8.1               [cpuonly]
* `tensorboard`               2.3.0
* `numpy`                     1.18.5

First install conda with Python 3.8. To install packages (only need the following):
```
conda install pytorch torchvision cpuonly -c pytorch
pip install tensorboard
```

To run python scripts in the directories above, export the following variable name when a new terminal is started:
```
export PYTHONPATH="${PYTHONPATH}:<path to>/ActiveBinaryClassification
```