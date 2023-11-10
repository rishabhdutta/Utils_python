import numpy as np

def remove_plane(in_data, opt=0, xv=None, yv=None):
    lin, col = in_data.shape

    if xv is None or yv is None:
        x, y = np.meshgrid(np.arange(1, col + 1), np.arange(1, lin + 1))
    else:
        x, y = np.meshgrid(xv, yv)

    if opt == 0:
        G = np.column_stack([np.ones_like(x.flatten()), x.flatten(), y.flatten()])
        G2 = np.column_stack([np.ones_like(x.flatten()), x.flatten(), y.flatten()])
    elif opt == 1:
        G = np.column_stack([np.ones_like(x.flatten()), x.flatten(), y.flatten(), x.flatten() * y.flatten(),
                             x.flatten() ** 2, y.flatten() ** 2])
        G2 = np.column_stack([np.ones_like(x.flatten()), x.flatten(), y.flatten(), x.flatten() * y.flatten(),
                              x.flatten() ** 2, y.flatten() ** 2])
    else:
        raise ValueError("Invalid value for 'opt'. It should be 0 or 1.")

    d0 = in_data.flatten()
    ind = ~np.isnan(d0)
    nnv = np.sum(ind)

    if nnv < 3:
        print('Too few non-nan points')
        return None, None, None, None

    G = G[ind, :]
    d = d0[ind]

    m = np.linalg.inv(G.T @ G) @ G.T @ d

    plane = np.matmul(G2, m).reshape(lin, col)

    drem = d0 - plane.flatten()

    out_data = np.reshape(drem, (lin, col))

    return out_data, plane, m, (x, y)