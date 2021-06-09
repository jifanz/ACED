import sys
import argparse
import os
import subprocess
import numpy as np
import time
from iwal import utils


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
opts = parser.parse_args()

if opts.data == 'cifar64':
    X_tr, Y_tr, X_te, Y_te = utils.get_cifar64_data()
    utils.vw_format(X_tr, Y_tr, 'cifar_train')
    utils.vw_format(X_te, Y_te, 'cifar_test')
    train_set = 'cifar_train.vw'
    test_set = 'cifar_test.vw'
    train_set_passes = 'cifar_train_passes.vw'
elif opts.data == 'mnist_half':
    X_tr, Y_tr, X_te, Y_te = utils.get_MNISThalf()
    utils.vw_format(X_tr, Y_tr, 'mnist_train')
    utils.vw_format(X_te, Y_te, 'mnist_test')
    train_set = 'mnist_train.vw'
    test_set = 'mnist_test.vw'
    train_set_passes = 'mnist_train_passes.vw'
elif opts.data == 'svhn':
    X_tr, Y_tr, X_te, Y_te = utils.get_svhn_data()
    utils.vw_format(X_tr, Y_tr, 'svhn_train')
    utils.vw_format(X_te, Y_te, 'svhn_test')
    train_set = 'svhn_train.vw'
    test_set = 'svhn_test.vw'
    train_set_passes = 'svhn_train_passes.vw'
elif opts.data == 'fashion':
    X_tr, Y_tr, X_te, Y_te = utils.get_fashion_data()
    utils.vw_format(X_tr, Y_tr, 'fashion_train')
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

vw_initial_train = 'vw -f tmp/{} --binary --passes 1 --l2 0 --loss_function logistic'
vw_train = 'vw -f tmp/{} --binary --passes 1 --l2 0 --loss_function logistic'

vw_train_predict = 'vw -i tmp/{} -t -p tmp/pred_train_{}.txt --binary -d {}'
vw_test_predict = 'vw -i tmp/{} -t -p tmp/pred_test_{}.txt --binary -d {}'

tr_accs = []
te_accs = []

np.random.seed(123)

for _ in range(10):
    run_command('rm -r tmp/*')

    tr_idxs = np.random.permutation(X_tr.shape[0])
    if opts.data == 'cifar64':
        X_tr, Y_tr, X_te, Y_te = utils.get_cifar64_data()
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'cifar_train')
        utils.vw_format(X_te, Y_te, 'cifar_test')
        train_set = 'cifar_train.vw'
        test_set = 'cifar_test.vw'
        train_set_passes = 'cifar_train_passes.vw'
    elif opts.data == 'mnist_half':
        X_tr, Y_tr, X_te, Y_te = utils.get_MNISThalf()
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'mnist_train')
        utils.vw_format(X_te, Y_te, 'mnist_test')
        train_set = 'mnist_train.vw'
        test_set = 'mnist_test.vw'
        train_set_passes = 'mnist_train_passes.vw'
    elif opts.data == 'svhn':
        X_tr, Y_tr, X_te, Y_te = utils.get_svhn_data()
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'svhn_train')
        utils.vw_format(X_te, Y_te, 'svhn_test')
        train_set = 'svhn_train.vw'
        test_set = 'svhn_test.vw'
        train_set_passes = 'svhn_train_passes.vw'
    elif opts.data == 'fashion':
        X_tr, Y_tr, X_te, Y_te = utils.get_fashion_data()
        utils.vw_format(X_tr[tr_idxs], Y_tr[tr_idxs], 'fashion_train')
        utils.vw_format(X_te, Y_te, 'fashion_test')
        train_set = 'fashion_train.vw'
        test_set = 'fashion_test.vw'
        train_set_passes = 'fashion_train_passes.vw'

    train_data = open(train_set, 'r').readlines()
    train_labels = np.array([float(l.split(' ')[0]) for l in train_data])
    train_len = len(train_labels)

    # duplicate the data
    with open(train_set_passes, 'w') as f:
        for i in range(passes):
            for l in train_data:
                f.write(l)

    checkpoints = []
    num_trained = []
    tr_acc = []
    te_acc = []

    total_rounds = int(np.ceil(train_labels.shape[0] * passes / float(batch)))
    total_queries_taken = 0

    k = 0
    print('total_rounds', total_rounds)
    while k < total_rounds:
        start = 1
        end = min(batch * (k + 1), len(train_labels) * passes)
        num_trained.append(end)
        # TRAINING PHASE
        print('running phase', k, start, end)
        cmd = sed_maker(start, end, train_set_passes) + '|' + vw_initial_train.format('model{}.vw'.format(k), 1)
        print(cmd)
        a = run_command(cmd)
        queries_taken = end
        checkpoints.append(queries_taken)
        print('took', queries_taken)

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

    print(num_trained)
    print(checkpoints)
    print(tr_acc)
    print(te_acc)
    tr_accs.append(tr_acc)
    te_accs.append(te_acc)

with open('vw_passive_{}_{}.npy'.format(opts.data, opts.batch), 'wb') as f:
    np.save(f, [np.mean(np.array(tr_accs), axis=0), np.mean(np.array(te_accs), axis=0), num_trained, checkpoints])
