import sys
import argparse
import os
import subprocess
import numpy as np
import time
from iwal import utils
import re

def run_command(cmd):
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)
    a = p.stderr.readlines()
    return a


def sed_maker(start, end, file):
    assert start < end
    return "sed -n '{},{}p;{}q' {}".format(start, end, end, file)


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='dataset', type=str, choices=['cifar64', 'mnist_half', 'fashion', 'svhn'])
parser.add_argument('--batch', help='how often to save the model', type=int)
parser.add_argument('--passes', help='passes', type=int)
parser.add_argument('--c0', nargs='+', type=float, default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
parser.add_argument('--ora', default=False, action='store_true')
opts = parser.parse_args()
# TODO alpha and beta hyperparameters, see parsimonious appendix
# TODO learning rates

for c0 in opts.c0:
    run_command('rm -r tmp/*')

    print("=" * 20 + " C0 = {} ".format(c0) + "=" * 20)
    run_command("rm -r tmp/*")

    if opts.data == 'cifar64':
        X_tr, Y_tr, X_te, Y_te = utils.get_cifar64_data()
        tr_idxs = np.random.permutation(X_tr.shape[0])
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'cifar_train')
        utils.vw_format(X_te, Y_te, 'cifar_test')
        train_set = 'cifar_train.vw'
        test_set = 'cifar_test.vw'
        train_set_passes = 'cifar_train_passes.vw'
    elif opts.data == 'mnist_half':
        X_tr, Y_tr, X_te, Y_te = utils.get_MNISThalf()
        tr_idxs = np.random.permutation(X_tr.shape[0])
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'mnist_train')
        utils.vw_format(X_te, Y_te, 'mnist_test')
        train_set = 'mnist_train.vw'
        test_set = 'mnist_test.vw'
        train_set_passes = 'mnist_train_passes.vw'
    elif opts.data == 'svhn':
        X_tr, Y_tr, X_te, Y_te = utils.get_svhn_data()
        tr_idxs = np.random.permutation(X_tr.shape[0])
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'svhn_train')
        utils.vw_format(X_te, Y_te, 'svhn_test')
        train_set = 'svhn_train.vw'
        test_set = 'svhn_test.vw'
        train_set_passes = 'svhn_train_passes.vw'
    elif opts.data == 'fashion':
        X_tr, Y_tr, X_te, Y_te = utils.get_fashion_data()
        tr_idxs = np.random.permutation(X_tr.shape[0])
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'fashion_train')
        utils.vw_format(X_te, Y_te, 'fashion_test')
        train_set = 'fashion_train.vw'
        test_set = 'fashion_test.vw'
        train_set_passes = 'fashion_train_passes.vw'

    test_data = open(test_set, 'r').readlines()
    test_labels = [float(l.split(' ')[0]) for l in test_data]
    test_len = len(test_labels)

    train_data = open(train_set, 'r').readlines()
    train_labels = np.array([float(l.split(' ')[0]) for l in train_data])
    train_len = len(train_labels)

    batch = opts.batch
    passes = opts.passes

    # duplicate the data
    with open(train_set_passes, 'w') as f:
        for i in range(passes):
            for l in train_data:
                f.write(l)

    vw_initial_train = 'vw -f tmp/{} -P 1 --active_cover --mellowness={} --loss_function=logistic --binary --learning_rate {}' + (
        ' --oracular' if opts.ora else '')
    vw_train = 'vw -f tmp/{} -P 1 --active_cover --mellowness={} --loss_function=logistic --binary --learning_rate {}' + (
        ' --oracular' if opts.ora else '')

    vw_train_predict = 'vw -i tmp/{} -t --loss_function=logistic -p tmp/pred_train_{}.txt --binary {}'
    vw_test_predict = 'vw -i tmp/{} -t --loss_function=logistic -p tmp/pred_test_{}.txt --binary {}'

    checkpoints = []
    num_trained = []
    tr_acc = []
    te_acc = []

    total_rounds = train_labels.shape[0] * passes / batch + 1
    total_queries_taken = 0

    k = 0
    print('total_rounds', total_rounds)
    while k < total_rounds:
        start = 1  # batch*k+1
        end = min(batch * (k + 1), len(train_labels) * passes)
        num_trained.append(end)
        # TRAINING PHASE
        print('running phase', k, start, end)
        cmd = sed_maker(start, end, train_set_passes) + '|' + vw_initial_train.format('model{}.vw'.format(k), c0,
                                                                                      .5 if opts.data == 'cifar64' else 1)
        print(cmd)
        a = run_command(cmd)
        # try:
        #     counter = None
        #     for line in a:
        #         if len(line) >= 4 and line.decode()[:4] == 'loss':
        #             counter = []
        #             continue
        #         if counter is not None:
        #             if line.decode() == '\n':
        #                 break
        #             else:
        #                 lst = re.sub(' +', ' ', line.decode()).split(' ')
        #                 counter.append(int(lst[2]) - 1)
        #     queries_taken = len(np.unique(np.array(counter) % train_labels.shape[0]))
        # except:
        #     print('error')
        #     print(a[-1].decode())
        #     print(a)
        #     sys.exit()
        # checkpoints.append(queries_taken)
        # print('took', queries_taken)

        # TEST PHASE
        pred_test_file = 'tmp/pred_test_{}.txt'.format(k)
        pred_train_file = 'tmp/pred_train_{}.txt'.format(k)

        cmd_test = vw_test_predict.format('model{}.vw'.format(k), k, test_set)
        cmd_train = vw_train_predict.format('model{}.vw'.format(k), k, train_set)

        print(cmd_test)
        b_test = run_command(cmd_test)
        while not os.path.exists(pred_test_file):
            print('sleeping')
            time.sleep(1)
        predictions = np.array([int(x) for x in open(pred_test_file, 'r').readlines()])
        te_acc.append(np.sum(test_labels == predictions) / len(predictions))
        print('cmd line test error', b_test[-4], 'accuracy', te_acc[-1])

        b_train = run_command(cmd_train)
        while not os.path.exists(pred_train_file):
            print('sleeping')
            time.sleep(1)
        predictions = np.array([int(x) for x in open(pred_train_file, 'r').readlines()])
        tr_acc.append(np.sum(train_labels == predictions) / len(predictions))
        print('cmd line train error', b_train[-4], 'accuracy', tr_acc[-1])
        print('\n')
        k += 1

    # print(num_trained)
    # print(checkpoints)
    print(te_acc)
    print(tr_acc)

    run_command(cmd)
    time.sleep(1)

    checkpoints = np.zeros(len(tr_acc))
    with open("./queries.txt", "r") as f:
        idxs = []
        for line in f:
            idxs.append(int(line))
        print(len(idxs))
        j = 0
        queried = []
        for i in idxs:
            idx = i % len(train_labels)
            if idx not in queried:
                queried.append(idx)
                checkpoints[i // batch] += 1
    checkpoints = list(np.cumsum(checkpoints).astype(int))
    num_trained = checkpoints
    print(checkpoints)

    with open('vw_{}_{}_{}_{}{}.npy'.format(opts.data, opts.batch, opts.passes, c0, '_ora' if opts.ora else ''),
              'wb') as f:
        np.save(f, [tr_acc, te_acc, num_trained, checkpoints])
