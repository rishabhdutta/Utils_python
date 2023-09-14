import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy
from scipy.linalg import inv

def quadtree_level(oldind):
    '''
    Add a new quadtree partitioning level
    '''
    indexmatrix = []
    lin, col = oldind.shape
    nlin = 1

    for k in range(lin):
        if oldind[k, col - 4] == 1:
            tmp1 = np.concatenate((oldind[k, :col - 4], [0]))
            tmp2 = oldind[k, col - 3:]
            indexmatrix = np.vstack((indexmatrix, np.concatenate((tmp1, tmp2))))
            nlin += 1
        else:
            tmp1 = np.repeat([oldind[k, :col - 4]], 4, axis=0)
            tmp2 = np.column_stack((np.zeros(4), np.repeat([oldind[k, col - 2:]], 4, axis=0)))
            indexmatrix = np.vstack((indexmatrix, np.column_stack((tmp1, [1, 2, 3, 4]))))
            indexmatrix = np.vstack((indexmatrix, tmp2))
            nlin += 4

    return indexmatrix

def fit_bilinplane(data, coord):
    # Clean data and coordinates of NaNs, update coordinate vector accordingly:
    no_nan_ind = np.where(~np.isnan(data))[0]
    d = data[no_nan_ind]
    coord = coord[:, no_nan_ind]

    # Get the number of data points
    N = len(d)

    # If 3 or more data points are left after NaN screening, then
    if N >= 3:
        # Make matrix G
        ones = np.ones(N)
        G = np.vstack((ones, coord[0, :], coord[1, :])).T
        gtginv = inv(np.dot(G.T, G))

        m = np.dot(gtginv, np.dot(G.T, d))

        # Calculate the rms
        rootms = np.sqrt(np.mean((d - np.dot(G, m))**2))
    else:
        rootms = 0
        G = 0
        m = np.array([0, 0, 0])

    return m, G, rootms


