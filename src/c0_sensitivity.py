import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

mode = 'train'
data_name = 'cifar'
accum_fn = np.maximum.accumulate if mode == 'train' else lambda x: x

if data_name == 'cifar':
    dir = 'log_10000_64_cifar_binary'
    dataset = 'cifar64'

logs = []
accuracies = []

with open(dir + '/' + dataset + '_passive_sklogistic_passive_0.pkl', 'rb') as f:
    passive = pickle.load(f)

passive_accuracy = passive['accuracy' if mode == 'train' else 'test_accuracy']
passive_schedule = passive['schedule']
print(len(passive_accuracy))
plt.plot(passive_schedule, accum_fn(passive_accuracy), marker='o', ms=5, label='Passive', linewidth=3)

dir = '../vw'

if data_name == 'cifar':
    num_passes = 20
    C0 = [1e-3, 3e-3, 1e-2]
    for c0 in C0:
        print('{}/vw_cifar64_250_{}_{}.npy'.format(dir, num_passes, c0))
        with open('{}/vw_cifar64_250_{}_{}.npy'.format(dir, num_passes, c0), 'rb') as f:
            tr_acc, te_acc, num_trained, checkpoints = np.load(f, allow_pickle=True)
        plt.plot(np.array(checkpoints), accum_fn(tr_acc if mode == 'train' else te_acc),
                 marker='o', ms=5, label='OAC (VW), C0={}'.format(c0), linewidth=3)
plt.legend()
plt.xlabel("Number of Queries")
plt.ylabel("Accuracy")
plt.show()
