from src.dataset import *
from src.argmax_oracle import argmax_oracle_single
from src.record import Recorder
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


if __name__ == "__main__":
    np.random.seed(123)
    mp.set_start_method('spawn')

    shared_data = get_shared_data()
    init_worker(*shared_data)
    dataset = get_dataset()

    passive_writer = SummaryWriter('runs/passive')
    print("************ Starting Passive Run #%d *************" % 0)
    N_point = dataset["X"].size(0)
    schedule = list(np.arange(250, N_dim, step=250)) + [N_dim]
    N_checkpoint = len(schedule)
    N_run = 10
    recorder = Recorder(data_name, "passive_" + model_name, 'passive', 0)
    y = dataset["Y"].numpy()

    # Random shuffling of the dataset among N_run runs.
    idx = [torch.randperm(N_point) for _ in range(N_run)]
    print('Num checkpoints', N_checkpoint, int(np.ceil(N_point / float(N_checkpoint))))
    for i in range(1, N_checkpoint + 1):
        print(i)
        sum_accuracy = 0
        sum_test_accuracy = 0
        training_size = schedule[i - 1]
        for j in range(N_run):
            w = np.zeros(N_point, dtype=float)
            w[idx[j][:training_size]] = y[idx[j][:training_size]] * 2 - 1
            z, test_acc = argmax_oracle_single(w)
            accuracy = np.mean((y == z).astype(float))
            sum_accuracy += accuracy
            sum_test_accuracy += test_acc
        passive_writer.add_scalar('Accuracy/results/retrain', sum_accuracy / N_run, training_size)
        recorder.append_var("accuracy", sum_accuracy / N_run)
        passive_writer.add_scalar('Accuracy/results/retrain_test', sum_test_accuracy / N_run, training_size)
        recorder.append_var("test_accuracy", sum_test_accuracy / N_run)
    recorder.record_var("schedule", schedule)
    recorder.save()
    passive_writer.close()

