import os

os.environ["BLIS_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

import numpy as np
from numpy import sqrt, log
from iwal.utils import *
from sklearn.linear_model import LogisticRegression
import sys
import time
import warnings
import pickle
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import multiprocessing as mp
import argparse

import mkl
mkl.set_num_threads(1)


def get_passive(X_tr, Y_tr, X_te, Y_te, batch=100, trials=10):
    checkpoints = []
    acc_tr = []
    acc_te = []
    n_round = int(X_tr.shape[0] / batch) + 1
    perms = [np.random.permutation(X_tr.shape[0]) for _ in range(trials)]
    for i in range(1, n_round):
        print(i, n_round)
        checkpoints.append([])
        total_score = 0
        for j in range(trials):
            clf = LogisticRegression()
            final = min(batch * i, X_tr.shape[0] - 1)
            clf.fit(X_tr[perms[j]][:final], Y_tr[perms[j]][:final])
            checkpoints[-1].append(final)
            total_score += clf.score(X_tr, Y_tr)
        acc_tr.append(total_score / trials)
    return acc_tr, acc_te, checkpoints


def get_model(X, Y, w, iters=1000, b=True, tol=1e-3):
    clf = LogisticRegression(C=1e8, max_iter=iters, tol=tol, solver='lbfgs', fit_intercept=b)
    clf.fit(X, Y, sample_weight=w)
    return clf


def get_alternative(X, Y, x, y, w):
    clfa = get_model(X - x, Y, w, b=False)
    clfa.intercept_ += 1e-20 * (1 - y)
    return clfa


class IWAL():
    def __init__(self, X_tr, Y_tr, X_te, Y_te, c0=.01):
        print('starting training', c0)
        self.X_tr = X_tr
        self.X_te = X_te
        self.Y_tr = Y_tr
        self.Y_te = Y_te
        self.n = X_tr.shape[0]
        self.c0 = c0
        self.c1 = 5 + 2 * np.sqrt(2)
        self.c2 = 5
        self.errors = 0
        self.P = []
        self.G = []
        self.entropy = []

    def get_query(self, clf, idx, k, diagnostic=False, thresholdtype='iwal0'):
        x = (self.X_tr[idx])[np.newaxis]
        y = clf.predict(x)[0]

        clfa = get_alternative(self.X_tr[self.sk_idxs], self.Y_f, x, y, self.sk_w)
        G = (clf.score(self.X_tr[self.sk_idxs], self.Y_f, sample_weight=self.sk_w)
             - clfa.score(self.X_tr[self.sk_idxs] - x, self.Y_f, sample_weight=self.sk_w))
        assert clfa.predict(x - x)[0] == 1 - y, "pred: {}, y: {}".format(clfa.predict(x)[0], 1 - y)
        # print(G)
        if diagnostic:
            print('index', idx, 'actual', self.Y_tr[idx], 'predicted', y,
                  'G', G, -sum(clf.predict_log_proba(x)[0] * clf.predict_proba(x)[0]))
            self.entropy.append(-sum(clf.predict_log_proba(x)[0] * clf.predict_proba(x)[0]))

        self.G.append(G)

        c = self.c0 * log(k) / (k)
        if thresholdtype == 'iwal0':
            threshold = np.sqrt(c) + c
        else:
            threshold = np.sqrt(c * self.score) + c

        if G < threshold:
            P = 1
        else:
            if thresholdtype == 'iwal0':
                a, b = self.c1, -self.c1 + 1
            else:
                a, b = self.c1 * np.sqrt(self.score), (-self.c1 + 1) * np.sqrt(self.score)
            d, e = self.c2, -self.c2 + 1
            P = ((a * np.sqrt(c) + np.sqrt(a ** 2 * c + 4 * d * c * (G - b * np.sqrt(c) - e * c))) / (
                    2 * (G - b * np.sqrt(c) - e * c))) ** 2
        self.P.append(P)
        return P

    def run(self, nqueries, init_queries=100, batch=100, diagnostic=False, querystyle='reg', thresholdtype='iwal0'):
        # retraining
        self.acc_tr = []
        self.acc_te = []
        # weighted
        self.acc_model_tr = []
        self.acc_model_te = []
        self.checkpoints = []
        self.idxs = np.zeros(self.X_tr.shape[0], dtype=bool)
        idxs_seed = np.random.choice(np.arange(self.n), init_queries, replace=False)
        self.Y_f = [self.Y_tr[i] for i in idxs_seed]  # len num queries
        self.idxs[idxs_seed] = True
        self.sk_idxs = idxs_seed.tolist()  # len num queries
        self.sk_w = [1 for i in range(init_queries)]  # len num queries

        self.clf = get_model(self.X_tr[self.sk_idxs], self.Y_f, self.sk_w)
        self.score = self.clf.score(self.X_tr[self.sk_idxs], self.Y_f, sample_weight=self.sk_w)

        init_time = time.time()

        total_queries = init_queries  # how many we have paid for
        k = init_queries  # round
        while k < nqueries:
            k = k + 1
            idx = k % self.X_tr.shape[0]

            P = self.get_query(self.clf, idx, k, diagnostic=diagnostic, thresholdtype=thresholdtype)
            train_flag = False
            if (querystyle == 'reg' and np.random.rand() < P) or P == 1:  # condition to take a query
                train_flag = True
                self.sk_idxs.append(idx)
                self.sk_w.append(1 / P)
                self.Y_f.append(self.Y_tr[idx])
                if self.idxs[idx] == 0:
                    total_queries += 1
                    self.idxs[idx] = 1
                    if total_queries % batch == 0:
                        self.checkpoints.append(total_queries)
                        self.acc_model_tr.append(self.clf.score(self.X_tr, self.Y_tr))
                        self.acc_model_te.append(self.clf.score(self.X_te, self.Y_te))

                        clfr = get_model(self.X_tr[self.idxs], self.Y_tr[self.idxs],
                                         np.ones(np.sum(self.idxs.astype(int))))
                        self.acc_tr.append(clfr.score(self.X_tr, self.Y_tr))
                        self.acc_te.append(clfr.score(self.X_te, self.Y_te))
                        print('rounds', k, 'queries', total_queries)
                        print('train_accuracy', clfr.score(self.X_tr, self.Y_tr))
                        print('errors', self.errors, 'time', time.time() - init_time)
            elif querystyle == 'ora':
                train_flag = True
                self.sk_idxs.append(idx)
                self.sk_w.append(1)
                self.Y_f.append(self.clf.predict(self.X_tr[idx][np.newaxis])[0])
            if train_flag:
                self.clf = get_model(self.X_tr[self.sk_idxs], self.Y_f, self.sk_w)
                self.score = self.clf.score(self.X_tr[self.sk_idxs], self.Y_f, sample_weight=self.sk_w)


if __name__ == "__main__":
    print(np.__config__.show())
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='dataset', type=str)
    parser.add_argument('--initqueries', help='init number queries', type=int, default=250)
    parser.add_argument('--batch', help='number queries every checkpoint', type=int, default=250)
    parser.add_argument('--passes', help='number of passes', type=int, )
    parser.add_argument('--thresholdtype', help='threshold type', type=str, choices=['iwal0', 'iwal1'], )
    parser.add_argument('--querystyle', help='sampling style', type=str, choices=['reg', 'ora'], )
    parser.add_argument('--diagnostic', help='diagnostic', type=bool, default=False)
    parser.add_argument('--clist', nargs='+', type=float, default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    opts = parser.parse_args()

    if opts.data == 'cifar64':
        X_tr, Y_tr, X_te, Y_te = get_cifar64_data()
    elif opts.data == 'mnist_half':
        X_tr, Y_tr, X_te, Y_te = get_MNISThalf()
    elif opts.data == 'fashion':
        X_tr, Y_tr, X_te, Y_te = get_fashion_data()
    elif opts.data == 'svhn':
        X_tr, Y_tr, X_te, Y_te = get_svhn_data()
    else:
        raise Exception("Dataset not found.")

    C0 = opts.clist


    def run_single(opts, X_tr, Y_tr, X_te, Y_te, c0, i):
        np.random.seed(123)
        s = IWAL(X_tr, Y_tr, X_te, Y_te, c0=c0)
        s.run(nqueries=opts.passes * X_tr.shape[0], init_queries=opts.initqueries, diagnostic=opts.diagnostic,
              querystyle=opts.querystyle, thresholdtype=opts.thresholdtype, batch=opts.batch)
        name = 'log/{}_{}_{}_{}'.format(opts.data, int(-np.log10(c0)), opts.querystyle, opts.thresholdtype)
        with open(name + '.pkl', 'wb') as f:
            pickle.dump((s, c0), f)
        return s


    print(opts)
    pool = mp.Pool(len(C0))
    final = pool.starmap(run_single, [(opts, X_tr, Y_tr, X_te, Y_te, c0, i) for i, c0 in zip(range(len(C0)), C0)])
