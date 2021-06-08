import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d, LinearNDInterpolator

dir = 'log'
dataset = '2d_sample'

with open('../data/2d_sample_3.pkl', 'rb') as file:
    X_data, Y_data, X_test_data, Y_test_data = pickle.load(file)

runs = [4]
for i in runs:
    with open(dir + '/' + dataset + '_2d_thresh_combi_alg_global_{}.pkl'.format(i), 'rb') as f:
        alloc = pickle.load(f)['lk'][0]

    # f = interp2d(X_data[:, 0], X_data[:, 1], np.array(alloc), kind="linear")

    x_coords = np.linspace(0, 1, 5000)
    z_coords = np.linspace(0, 1, 5000)
    x_coords, z_coords = np.meshgrid(x_coords, z_coords)

    interp = LinearNDInterpolator(X_data, alloc)

    interped = interp(x_coords, z_coords)

    fig = plt.imshow(interped,  #  f(x_coords, z_coords),
               extent=[np.min(x_coords), np.max(x_coords), np.min(z_coords), np.max(z_coords)],
               origin="lower", interpolation=None)
    plt.colorbar()

    plt.scatter(X_data[:, 0], X_data[:, 1], c=['r' if l==1 else 'y' for l in Y_data], s=5)

    # Show the positions of the sample points, just to have some reference
    # fig.axes.set_autoscale_on(False)
    # plt.scatter(X_data[:, 0], X_data[:, 1], 400, facecolors='none')
    plt.show()

    # plt.scatter(X_data[alloc > 0.004][:, 0], X_data[alloc > 0.004][:, 1], s=10)
    # plt.show()
    #
    # plt.scatter(X_data[alloc < 0.002][:, 0], X_data[alloc < 0.002][:, 1], s=10)
    # plt.show()
