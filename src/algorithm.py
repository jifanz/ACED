'''
Main implementation of our algorithm.

Mappings of variable names to paper:
    z0: vector of dimension N_dim, tilde{h_k} in the paper.
    l_inv: vector of dimension N_dim, 1 / lambda in the paper.
    shift: a non-negative number, 2^{-k+1} in the paper.
    theta: vector of dimension N_dim, hat{eta_{k - 1}} in the paper.
    eta: vector of dimension N_dim, zeta in the paper.
'''
import logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.argmax_oracle import argmax_oracle_single
from src.dataset import *
from src.utils import *
from src.record import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def maxZ(z0, l_inv, shift, theta, eta, num_grid=40):
    '''
    Computes
    max_h f(lambda; h; zeta)

    Input:
        z0: vector of dimension N_dim, tilde{h_k} in the paper.
        l_inv: vector of dimension N_dim, 1 / lambda in the paper.
        shift: a non-negative number, 2^{-k+1} in the paper.
        theta: vector of dimension N_dim, hat{eta_{k - 1}} in the paper.
        eta: vector of dimension N_dim, zeta in the paper.
        tol: stopping criteria for search.

    Output:
        z: argmax vector, denoting the prediction of the hypothesis.
        val: value of max_h f(lambda; h; zeta)
    '''
    seed = np.random.randint(1, 1e8)

    def glinmax(r, return_pred=False, debug=False):
        v = -np.sqrt(l_inv) * eta + r * theta
        z = argmax_oracle_single(v, seed=seed, return_test_accuracy=False)
        if z @ theta > z0 @ theta:
            # Found better hypothesis than z0, so we replace it with the new hypothesis.
            # This is necessary since we are using an approximate oracle where we relax the 0/1 loss.
            raise ZException('Found Better', z)
        val = z @ v
        diff = val - r * (shift + theta @ z0) + np.sum(z0 * np.sqrt(l_inv) * eta)
        frac = np.sum((z0 - z) * np.sqrt(l_inv) * eta) / (shift + theta @ (z0 - z))

        if return_pred:
            return diff, frac, z
        return diff, frac

    r = 100
    factor = 10
    rs = [r]
    diffs = []
    fracs = []
    zs = []

    diff, frac, z = glinmax(r, return_pred=True)
    diffs.append(diff)
    fracs.append(frac)
    zs.append(z)
    it = 0
    while diff < 0 and it < num_grid:
        r /= 2
        diff, frac, z = glinmax(r, return_pred=True)
        rs.append(r)
        diffs.append(diff)
        fracs.append(frac)
        zs.append(z)
        it += 1

    for _ in range(num_grid - it):
        diff, frac, z = glinmax(r, return_pred=True)
        rs.append(r)
        diffs.append(diff)
        fracs.append(frac)
        zs.append(z)
        if diff > 0:
            r *= factor
        else:
            r /= factor ** 2
            factor /= 1.4
    idx = np.argmax(np.array(fracs))

    record = {'diffs': diffs, 'fracs': fracs, 'rs': rs}
    return None, zs[idx], record


def mp_batch(maxz_params):
    '''
    Wrapper around maxZ that also records information for logging.

    Input:
        maxz_params: a dictionary from function argument name to values to be passed to maxZ.
    '''
    _, z, record = maxZ(**maxz_params)
    record = {}
    z0 = maxz_params['z0']
    l_inv = maxz_params['l_inv']
    shift = maxz_params['shift']
    theta_k = maxz_params['theta']
    eta = maxz_params['eta']
    gamma_val = np.sum((z0 - z) * np.sqrt(l_inv) * eta) / (shift + (z0 - z) @ theta_k)
    record['gamma_val'] = gamma_val
    return gamma_val, z, eta, record


def eval_grad(z0, l_inv, shift, theta_k, eta_batch, pool):
    '''
    Computes
    E [max_h f(lambda; h; zeta)]
    where expectation is taken by averaging over eta_batch.shape[0] samples.

    Input:
        z0: vector of dimension N_dim, tilde{h_k} in the paper.
        l_inv: vector of dimension N_dim, 1 / lambda in the paper.
        shift: a non-negative number, 2^{-k+1} in the paper.
        theta_k: vector of dimension N_dim, hat{eta_{k - 1}} in the paper.
        eta_batch: array of (M, N_dim), each row corresponds to a sample of zeta in the paper.
        pool: multiprocessing pool.

    Output:
        z: argmax vector, denoting the prediction of the hypothesis.
        val: value of max_h f(lambda; h; zeta)
    '''
    finished = False
    while not finished:
        try:
            results = pool.map(mp_batch, [{'z0': z0, 'l_inv': l_inv, 'shift': shift, 'theta': theta_k,
                                           'eta': eta_batch[i, :], 'num_grid': 40}
                                          for i in range(eta_batch.shape[0])])
            finished = True
        except ZException as e:
            # Found better hypothesis than z0, so we replace it with the new hypothesis.
            # This is necessary since we are using an approximate oracle where we relax the 0/1 loss.
            print('resetted')
            z0 = e.z

    # Retrieve from results of parallel computation.
    zs = []
    records = []
    gamma_vals = []
    for (gamma_val, z, eta, record) in results:
        gamma_vals.append(gamma_val)
        zs.append(z)
        records.append(record)
    gamma_vals = np.array(gamma_vals)
    zs = np.array(zs)

    # Compute gradient wrt lambda.
    grads = -1 / 2 / np.sqrt(N_dim) * (z0 - zs) * np.sqrt(l_inv) ** 3 * eta_batch / (
            shift + (z0 - zs) @ theta_k.reshape((-1, 1)))

    avg_val = np.mean(gamma_vals)
    avg_val_var = np.mean(gamma_vals ** 2) - avg_val ** 2
    avg_g = np.mean(grads, axis=0)
    avg_g_var = np.mean(grads ** 2, axis=0) - avg_g ** 2
    return avg_val, avg_val_var / eta_batch.shape[0], avg_g, avg_g_var / eta_batch.shape[0], z0, records, zs


def gamma_combi(z0, theta_k, k, B, shared_data, iters=N_it_gamma_combi, max_batch_size=800, min_batch_size=50,
                max_lr=.1, min_lr=1e-5, eps_rel=0.2, eps_abs=1., visualize=False, recorder=None, trial=0, l=None):
    '''
    A function to minimize gamma* as a function of lambda.

    Input:
        z0: vector of dimension N_dim, tilde{h_k} in the paper.
        theta_k: vector of dimension N_dim, hat{eta_{k - 1}} in the paper.
        k: index number of iteration, k in the paper.
        B: scaling factor for 2^{-k}, usually 2 * N_dim to match paper.

        shared_data: dataset object to be passed into multiprocessing pool.

        iters: total number of iterations (100 should be plenty for all purposes).

        Batchsize (number of etas per iteration) is set adaptively in the algorithm. But will never exceed limits.
        min_batch_size: can be as small as 1, default 10. Let algorithm grow it.
        max_batch_size: should be a bit larger than gamma*. So maybe take this to be twice the total measurement budget
            you plan to take.

        Step size eta is set adaptively in the algorithm. But will never exceed limits.
        min_eta: in my experiments just on thresholds, .001 was small enough and almost never used.
        max_eta: in my experiments just on thresholds, 1.0 worked but could cause some lambdas to get suuuuuper small
            which is not great. 0.1 seems good.

        Stopping criteria. If max_batch_size is not large enough, or min_eta is not small enough, these stopping
            conditions may not be met.
        eps_rel: Uses confidence intervals to Stop when (Gamma(lambda) - Gamma(opt)) / Gamma(lambda) < eps_rel. That is,
            Gamma(lam) < Gamma(opt)/(1 - eps_rel). I suggest a conservative value like eps_rel=0.2.
        eps_abs: Uses confidence intervals to Stop when Gamma(lambda) - Gamma(opt) < eps_abs. This should be no smaller
            than 1, but could be quite large, like 10.
    '''
    shift = 2. ** (-k) * B
    if l is None:
        l = np.ones(N_dim) / N_dim
    kappa = np.sqrt(2 * np.log(10))

    num_resets = 0
    batch_size = min_batch_size
    lr_candidates = 1. / (10 ** (np.arange(int(-np.log10(max_lr)), int(-np.log10(min_lr)) + 1)))
    gamma_expectations = []

    pool = mp.Pool(N_max_processes, initializer=init_worker, initargs=shared_data)

    writer = SummaryWriter('runs/active_{}'.format(trial))

    l_inv = 1 / np.clip(l, a_min=1e-8, a_max=None)
    while True:
        eta_batch = np.random.randn(batch_size, N_dim)
        avg_val, avg_val_var, avg_g, avg_g_var, z0, records, zs = eval_grad(z0, l_inv, shift, theta_k, eta_batch, pool)
        subopt_gap = np.dot(avg_g, l) - np.min(avg_g)
        if subopt_gap < kappa * np.sqrt(max(avg_g_var)) and 2 * batch_size <= max_batch_size:
            batch_size *= 2
            print("New batch size:", batch_size)
        else:
            break

    grads = []
    grad_norms = []
    for t in range(iters):
        old_z0 = z0
        if recorder is not None:
            recorder.set_level('maxZ_{}'.format(t))

        subopt_gap = avg_g @ l - np.min(avg_g)
        grads.append(avg_g)
        grad = avg_g - np.min(avg_g)
        grad_norms.append(np.linalg.norm(avg_g - np.mean(avg_g)))

        flag = True
        for j, lr in enumerate(lr_candidates):
            eta_batch = np.random.randn(batch_size, N_dim)
            lbd = l * np.exp(-lr * grad)
            lbd = np.clip(lbd, a_min=1e-8, a_max=1)
            lbd /= np.sum(lbd)
            l_inv = 1 / np.clip(lbd, a_min=1e-8, a_max=None)
            if np.sum(np.isnan(l_inv)):
                continue
            avg_val, avg_val_var, avg_g, avg_g_var, z0, records, zs = eval_grad(z0, l_inv, shift, theta_k, eta_batch,
                                                                                pool)
            if flag or avg_val + np.sqrt(avg_val_var) < best_tuple[0] - np.sqrt(best_tuple[1]):
                flag = False
                best_tuple = (avg_val, avg_val_var, avg_g, avg_g_var, lbd, records, zs)
            else:
                break

        avg_val, avg_val_var, avg_g, avg_g_var, l, records, zs = best_tuple

        if subopt_gap < kappa * np.sqrt(np.max(avg_g_var)):
            if 2 * batch_size <= max_batch_size:
                batch_size *= 2

        # Logging
        gamma_expectations.append(avg_val)
        if (t + 1) >= iters // 10 and (t + 1) % (iters // 10) == 0:
            logging.info('gamma_combi: iter {}'.format(t + 1))

        writer.add_scalar('Gamma_Combi_{}/gradient_norms'.format(k), np.linalg.norm(avg_g - np.mean(avg_g)), t)
        writer.add_scalar('Gamma_Combi_{}/gamma_expectations'.format(k), gamma_expectations[-1], t)

        if recorder is not None:
            recorder.record_vars(['zs', 'record', 'etas'], [zs, records, eta_batch])
            recorder.pop()

        # Found better hypothesis, don't terminate.
        if np.sum(z0 != old_z0) != 0:
            continue

        # Relative suboptimality gap stopping condition.
        if (subopt_gap + kappa * np.sqrt(np.max(avg_g_var))) / (avg_val - kappa * np.sqrt(avg_val_var)) < eps_rel:
            logging.info('Gamma_Combi finished after {} iterations'.format(t + 1))
            break

        # Absolute suboptimality gap stopping condition.
        if subopt_gap + kappa * np.sqrt(np.max(avg_g_var)) < eps_abs:
            logging.info('Gamma_Combi finished after {} iterations'.format(t + 1))
            break

    pool.close()
    pool.join()

    for i in range(len(l)):
        writer.add_scalar('Gamma_Combi_{}/allocation'.format(k), l[i], i)

    if recorder is not None:
        recorder.record_vars(['ls', 'gamma_expectations', 'grads', 'grad_norms', 'num_resets'],
                             [l, gamma_expectations, grads, grad_norms, num_resets])

    writer.close()
    return l


def get_z0(theta_k, shared_data, num=100):
    '''
    More robust computation of argmax h <w, h>.
    '''
    pool = mp.Pool(N_max_processes, initializer=init_worker, initargs=shared_data)
    result_lst = pool.map(argmax_oracle_single, [theta_k] + [
        theta_k * np.random.choice([-1, 1], size=theta_k.shape[0], replace=True, p=[.01, .99]) for _ in range(num - 1)])
    pool.close()
    pool.join()
    z0_lst = [result[0] for result in result_lst]
    return result_lst[np.argmax(np.array(z0_lst) @ theta_k)]


def combi_alg(theta_star, z_star, schedule, it_run=0):
    '''
    Runs ACED.

    Input:
        theta_star: a vector of dimension N_dim with elements in {-1, 1}, 2 * h_star - 1.
        z_star: a vector of dimension N_dim with elements in {0, 1}, h_star in paper.
        schedule: number of new queries to take for each iteration.
        it_run: index the number of runs of the algorithm (for averaging over multiple runs).

    Output:
        z0: best arm
        lk: allocation
        thetak: empirically computed theta estimate
        np.sum(pulls): total number of queries
    '''
    shared_data = get_shared_data()
    init_worker(*shared_data)
    theta_k, theta_sum = 2 * np.random.rand(N_dim) - 1, 0
    B = 2 * N_dim
    pulls = np.zeros(N_dim)
    labels = np.zeros(N_dim)
    lk = np.ones(N_dim, dtype=float) / N_dim

    writer = SummaryWriter('runs/active_{}'.format(it_run))
    global_record = Recorder(data_name, model_name, f'combi_alg_global', idx=it_run)

    z0, _ = get_z0(theta_k, shared_data)
    prob_sum = np.zeros(N_dim, dtype=float)
    pk = lk

    for t in range(1, N_dim + 1):
        if t in schedule:
            k = schedule.index(t)
            recorder = Recorder(data_name, model_name, f'combi_alg_run_{it_run}', idx=k)
            logging.info('combi_alg: entering gamma_combi in round {}'.format(k))
            recorder.set_level('gamma_combi')
            lk = gamma_combi(z0, theta_k, k, B, shared_data, iters=N_it_gamma_combi, recorder=recorder,
                             trial=it_run, l=lk, min_batch_size=125, max_batch_size=2000)
            logging.info('combi_alg: got gamma')
            recorder.pop()
            recorder.save()
            del recorder

            # Water filling.
            p_left = 1.0
            if k != len(schedule) - 1:
                p_diff = (lk * (schedule[k + 1] - 1) - prob_sum) / (schedule[k + 1] - t)
            diff_sorted = -np.sort(-p_diff)
            pk = np.zeros(N_dim)
            for ind in range(N_dim - 1):
                diff = diff_sorted[ind] - diff_sorted[ind + 1]
                if p_left > diff * (ind + 1):
                    pk[p_diff >= diff_sorted[ind]] += diff
                    p_left -= diff * (ind + 1)
                else:
                    pk[p_diff >= diff_sorted[ind]] += p_left / (ind + 1)
                    break
            if np.sum(pk) < 1:
                pk = pk + (1. - np.sum(pk)) / N_dim
            if not np.allclose(np.sum(pk), 1):
                print(p_diff)
                print(diff_sorted)
                print(lk)
                print(pk)
            assert np.allclose(np.sum(pk), 1), "pk not summing to 1 (sum = {}), {}, {}".format(np.sum(pk), lk, pk)
            pk += 1e-8
            pk /= np.sum(pk)
        prob_sum += pk

        # Query based on pk.
        idx = np.random.choice(np.arange(N_dim)[pulls == 0], 1, p=pk[pulls == 0] / np.sum(pk[pulls == 0]))
        labels[idx] = theta_star[idx]
        pulls[idx] = 1
        theta_k = labels

        # Logging.
        if t in schedule:
            for ind in range(N_dim):
                writer.add_scalar('Gamma_Combi_{}/water_filling'.format(k), pk[ind], ind)
            logger.info(
                'combi_alg: total pulls up to this round {}, total seen this round {}'.format(sum(pulls), t))
            logger.info('combi_alg: positive labels seen {}/{}'.format(sum(labels == 1), sum((theta_star + 1) / 2)))

            # Update the best estimated hypothesis so far.
            z0, test_accuracy_z0 = get_z0(theta_k, shared_data)
            pred, test_accuracy_retrain = argmax_oracle_single(labels)
            accuracy_retrain = np.sum(pred == z_star) / float(N_dim)
            accuracy_z0 = np.sum(z0 == z_star) / float(N_dim)
            writer.add_scalar('Accuracy/results/retrain', accuracy_retrain, t)
            writer.add_scalar('Accuracy/results/z0', accuracy_z0, t)
            writer.add_scalar('Accuracy/results/retrain_test', test_accuracy_retrain, t)
            writer.add_scalar('Accuracy/results/z0_test', test_accuracy_z0, t)
            print("Accuracy after round %d: retrain %f   z0: %f" % (k, accuracy_retrain, accuracy_z0))

            global_record.append_vars(
                ['lk', 'pk', 'pulls', 'theta_k', 'z0', 'pred', 'accuracy_retrain', 'accuracy_z0', 'z_star',
                 'theta_star', 'labels', 'test_accuracy_retrain', 'test_accuracy_z0'],
                [lk, pk, np.array(pulls), theta_k, z0, pred, accuracy_retrain, accuracy_z0, z_star, theta_star,
                 labels, test_accuracy_retrain, test_accuracy_z0])
            global_record.save()
        elif (t + 1) in schedule:
            z0, _ = get_z0(theta_k, shared_data)

    global_record.record_var("schedule", schedule)
    global_record.save()
    writer.close()
    return z0, lk, theta_k, np.sum(pulls)


if __name__ == "__main__":
    np.random.seed(111)
    mp.set_start_method('spawn')

    shared_data = get_shared_data()
    init_worker(*shared_data)
    dataset = get_dataset()

    for trial in range(1):
        print("************ Starting Active Run #%d *************" % trial)
        z_star = dataset["Y"].numpy()
        result = combi_alg(z_star * 2 - 1, z_star, list(np.arange(250, N_dim, step=250)) + [N_dim], it_run=trial)
