import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import itertools

matplotlib.rcParams.update({'font.size': 15})

for mode in ['train', 'test']:

    accum_fn = np.maximum.accumulate if mode == 'train' else lambda x: x

    for data_name in ['cifar', 'svhn', 'fashion', 'mnist']:

        if data_name == 'cifar':
            dir = 'log_10000_64_cifar_binary'
            dataset = 'cifar64'
        elif data_name == 'svhn':
            dir = 'log_12160_512_svhn_binary'
            dataset = 'svhn'
        elif data_name == 'fashion':
            dir = 'log_12000_512_fashion_binary'
            dataset = 'fashion'
        elif data_name == 'mnist':
            dir = 'log'
            dataset = 'mnist_half'

        logs = []
        accuracies = []
        if data_name == 'cifar':
            runs = [0, 1, 2, 3, 4]
        elif data_name == 'mnist':
            runs = [1]
        else:
            runs = [0]
        for i in runs:
            with open(dir + '/' + dataset + '_sklogistic_combi_alg_global_{}.pkl'.format(i), 'rb') as f:
                logs.append(pickle.load(f))
            accuracies.append(logs[-1]['accuracy_retrain' if mode == 'train' else 'test_accuracy_retrain'])
            active_schedule = 250 * np.arange(1, len(accuracies[0]) + 1)
            print(len(logs[-1]['accuracy_retrain' if mode == 'train' else 'test_accuracy_retrain']))

        with open(dir + '/' + dataset + '_passive_sklogistic_passive_0.pkl', 'rb') as f:
            passive = pickle.load(f)

        passive_accuracy = passive['accuracy' if mode == 'train' else 'test_accuracy']
        passive_schedule = passive['schedule']
        print(len(passive_accuracy))

        active_accuracy = np.mean(np.array(accuracies), axis=0)

        if mode == 'train':
            plt.axhline(y=max(max(active_accuracy), max(passive_accuracy)), color=(.4, .4, .4), linestyle='--',
                        linewidth=2)

        dir = "../iwal/log"

        if data_name == 'cifar':
            algos = []
            algos = algos + ["{}_{}_ora_iwal0.pkl".format(dataset, i) for i in range(3, 4)]
            algos = algos + ["{}_{}_ora_iwal1.pkl".format(dataset, i) for i in range(3, 4)]
            algos = algos + ["{}_{}_reg_iwal0.pkl".format(dataset, i) for i in range(6, 7)]
            algos = algos + ["{}_{}_reg_iwal1.pkl".format(dataset, i) for i in range(6, 7)]

        if data_name == 'svhn':
            algos = []
            algos = algos + ["{}_{}_ora_iwal0.pkl".format(dataset, i) for i in range(3, 4)]
            algos = algos + ["{}_{}_ora_iwal1.pkl".format(dataset, i) for i in range(3, 4)]
            algos = algos + ["{}_{}_reg_iwal0.pkl".format(dataset, i) for i in range(6, 7)]
            algos = algos + ["{}_{}_reg_iwal1.pkl".format(dataset, i) for i in range(6, 7)]

        if data_name == 'fashion':
            algos = []
            algos = algos + ["{}_{}_ora_iwal0.pkl".format(dataset, i) for i in range(7, 8)]
            algos = algos + ["{}_{}_ora_iwal1.pkl".format(dataset, i) for i in range(7, 8)]
            algos = algos + ["{}_{}_reg_iwal0.pkl".format(dataset, i) for i in range(7, 8)]
            algos = algos + ["{}_{}_reg_iwal1.pkl".format(dataset, i) for i in range(7, 8)]

        if data_name == 'mnist':
            algos = []
            algos = algos + [None]
            algos = algos + [None]
            algos = algos + ["{}_{}_reg_iwal0.pkl".format(dataset, i) for i in range(5, 6)]
            algos = algos + ["{}_{}_reg_iwal1.pkl".format(dataset, i) for i in range(6, 7)]

        algo_names = algos

        marker = itertools.cycle(('x', '<', '>', 's', 'P', 'd', '*'))
        space = itertools.cycle((10, 10, 10, 10, 1, 1, 10))

        from iwal.run_iwal import IWAL

        for algo, name in zip(algos, algo_names):
            if algo is not None:
                accuracies = []
                print('{}/{}'.format(dir, algo))
                with open('{}/{}'.format(dir, algo), 'rb') as f:
                    iwal = pickle.load(f)
                if mode == 'train':
                    accuracies.append(iwal[0].acc_tr)
                else:
                    accuracies.append(iwal[0].acc_te)
                baseline_accuracy = np.mean(np.array(accuracies), axis=0)
                baseline_accuracy_err = 1.96 * np.sqrt(np.var(np.array(accuracies), axis=0))
                print(baseline_accuracy.shape, baseline_accuracy_err.shape)
                curve_name = name.split('_')
                if curve_name[0] == 'mnist':
                    curve_name = '{}, {}, C0={}'.format(curve_name[3], curve_name[4][:-4], iwal[1])
                else:
                    curve_name = '{}, {}, C0={}'.format(curve_name[2], curve_name[3][:-4], iwal[1])
                curve_name = curve_name.upper()
                baseline_schedule = iwal[0].checkpoints
                plt.plot(baseline_schedule, accum_fn(baseline_accuracy), marker=next(marker), ms=7,
                         markevery=next(space), label=curve_name, linewidth=1)
            else:
                ax = plt.axes()
                ax_color_cycle = ax._get_lines.prop_cycler
                next(ax_color_cycle)['color']
                next(marker)
                next(space)

        plt.plot(active_schedule, accum_fn(active_accuracy), marker=next(marker), ms=7, markevery=next(space),
                 label='ACED', linewidth=1)

        plt.plot(passive_schedule, accum_fn(passive_accuracy), marker=next(marker), ms=7, markevery=next(space),
                 label='Passive', linewidth=1)

        dir = '../vw'

        if data_name == 'cifar':
            num_passes = 20
            C0 = 3 * 10. ** np.arange(-3, -2)
            for c0 in C0:
                print('{}/vw_cifar64_250_{}_{}.npy'.format(dir, num_passes, c0))
                with open('{}/vw_cifar64_250_{}_{}.npy'.format(dir, num_passes, c0), 'rb') as f:
                    tr_acc, te_acc, num_trained, checkpoints = np.load(f, allow_pickle=True)
                plt.plot(np.array(checkpoints), accum_fn(tr_acc if mode == 'train' else te_acc), marker=next(marker),
                         ms=7, markevery=next(space), label='OAC (VW), C0={}'.format(c0), linewidth=1)

        if data_name == 'svhn':
            num_passes = 20
            C0 = 3 * 10. ** np.arange(-2, -1)
            for c0 in C0:
                print('{}/vw_svhn_250_{}_{}.npy'.format(dir, num_passes, c0))
                with open('{}/vw_svhn_250_{}_{}.npy'.format(dir, num_passes, c0), 'rb') as f:
                    tr_acc, te_acc, num_trained, checkpoints = np.load(f)
                plt.plot(np.array(checkpoints), accum_fn(tr_acc if mode == 'train' else te_acc), marker=next(marker),
                         ms=7, markevery=next(space), label='OAC (VW), C0={}'.format(c0), linewidth=1)

        if data_name == 'fashion':
            num_passes = 20
            C0 = [.1]
            for c0 in C0:
                print('{}/vw_fashion_250_{}_{}.npy'.format(dir, num_passes, c0))
                with open('{}/vw_fashion_250_{}_{}.npy'.format(dir, num_passes, c0), 'rb') as f:
                    tr_acc, te_acc, num_trained, checkpoints = np.load(f)
                plt.plot(np.array(checkpoints), accum_fn(tr_acc if mode == 'train' else te_acc), marker=next(marker),
                         ms=7, markevery=next(space), label='OAC (VW), C0={}'.format(c0), linewidth=1)

        if data_name == 'mnist':
            num_passes = 5
            C0 = [.003]
            for c0 in C0:
                print('{}/vw_mnist_half_250_{}_{}.npy'.format(dir, num_passes, c0))
                with open('{}/vw_mnist_half_250_{}_{}.npy'.format(dir, num_passes, c0), 'rb') as f:
                    tr_acc, te_acc, num_trained, checkpoints = np.load(f)
                plt.plot(np.array(checkpoints), accum_fn(tr_acc if mode == 'train' else te_acc), marker=next(marker),
                         ms=7, markevery=next(space), label='OAC (VW), C0={}'.format(c0), linewidth=1)
            plt.ylim([.75, None])

        plt.legend()
        plt.xlabel("Number of Queries")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.show()
