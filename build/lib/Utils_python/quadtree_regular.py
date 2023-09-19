import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy
from scipy.linalg import inv
from scipy.optimize import curve_fit

# written in matlab by Sigurjon Jonsson 2000 
# converted to python by Rishabh Dutta 2023

def quadtree_level(oldind):
    '''
    Add a new quadtree partitioning level
    '''
    lin, col = oldind.shape
    nlin = 1
    
    # Loop over every old quadtree partition
    for k in range(lin):
        
        if oldind[k, col - 4] == 1:  # If deeper part isn't needed, we add a 0
            tmp1 = np.concatenate((oldind[k, :col - 4], [0]))
            tmp2 = oldind[k, col - 4:]
            
            add_indexmatrix = np.column_stack((tmp1, tmp2))
            nlin += 1
        else:  # Deeper partition needed, we add three new lines to the matrix
            tmp1 = np.column_stack((np.tile(oldind[k, :col - 4], (4, 1)), np.array([1, 2, 3, 4])))
            tmp2 = np.column_stack((np.zeros((4, 1)), np.tile(oldind[k, col - 3:], (4, 1))))

            add_indexmatrix = np.column_stack((tmp1, tmp2))
            nlin += 4

        if k == 0: 
            indexmatrix = np.empty((0, add_indexmatrix.shape[1]))
            
        indexmatrix = np.vstack((indexmatrix, add_indexmatrix))
        
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


def getchunck(index, data):
    # Get length of real index values, the last three values are
    # to be assigned with the median or something else, the fourth
    # last value is the 'check' signal
    length = len(index) - 4

    # Get size of data
    lin, col = data.shape

    # Get number of lines, or blocksize
    blcksz = lin

    # Initialize
    lst = 0
    cst = 0

    # Loop over every column of the 'real' part of the index matrix
    for k in range(length):
        blcksz = blcksz // 2
        if index[k] == 1:
            lst = lst
            cst = cst
        elif index[k] == 2:
            lst = lst
            cst = cst + blcksz
        elif index[k] == 3:
            lst = lst + blcksz
            cst = cst + blcksz
        elif index[k] == 4:
            lst = lst + blcksz
            cst = cst

    # Pick out the chunk:
    chunk = data[lst:lst + blcksz, cst:cst + blcksz]

    return chunk

def plot_quadtree(indexmatrix, data):
    coord = []
    coordl = []
    cx = []
    cy = []
    
    lin, col = data.shape
    len_indexmatrix = len(indexmatrix)
    level = len(indexmatrix[0]) - 4
    
    for k in range(len_indexmatrix):
        blcksz = lin
        lst = 0
        cst = 0
        
        for l in range(level):
            if indexmatrix[k, l] != 0:
                blcksz = blcksz / 2
                if indexmatrix[k, l] == 1:
                    lst = lst
                    cst = cst
                elif indexmatrix[k, l] == 2:
                    lst = lst
                    cst = cst + blcksz
                elif indexmatrix[k, l] == 3:
                    lst = lst + blcksz
                    cst = cst + blcksz
                elif indexmatrix[k, l] == 4:
                    lst = lst + blcksz
                    cst = cst
        
        coord.append([lst - 1 + blcksz / 2, cst - 1 + blcksz / 2])
        coordl.append([np.nan, np.nan])
        coordl.append([lst - 1, cst - 1])
        coordl.append([lst - 1, cst - 1 + blcksz])
        coordl.append([lst - 1 + blcksz, cst - 1 + blcksz])
        coordl.append([lst - 1 + blcksz, cst - 1])
        coordl.append([lst - 1, cst - 1])
        
        cx.append([cst - 1, cst - 1 + blcksz, cst - 1 + blcksz, cst - 1])
        cy.append([lst - 1, lst - 1, lst - 1 + blcksz, lst - 1 + blcksz])
    
    coord = np.array(coord)
    coordl = np.array(coordl)
    cx = np.array(cx)
    cy = np.array(cy)
    
    # Plot quadtree partitioning figures
    plt.figure()
    plt.plot(coord[:, 1], -coord[:, 0], '.')
    plt.title('Center-locations of quadtree squares')
    plt.axis('image')
    plt.axis([0, col, -lin, 0])
    plt.xlabel('x-coordinate (column #)')
    plt.ylabel('y-coordinate (-line #)')
    
    plt.figure()
    plt.plot(coordl[:, 1], -coordl[:, 0])
    plt.title('Quadtree squares')
    plt.axis('image')
    plt.axis([0, col, -lin, 0])
    plt.xlabel('x-coordinate (column #)')
    plt.ylabel('y-coordinate (-line #)')
    
    plt.show()

# Example usage:
# Replace 'indexmatrix' and 'data' with your data
# plot_quadtree(indexmatrix, data)

def check_quadtree(oldindmat, data, tolerance, fittype):
    ilin, icol = oldindmat.shape

    newindmat = oldindmat.copy()

    for k in range(ilin):
        if oldindmat[k, icol - 3] == 0:
            chunck = getchunck(oldindmat[k, :], data)
            c1, c2 = np.where(~np.isnan(chunck) | (chunck == 0))
            chunck = chunck.flatten()
            chunck_noNaN = chunck[c1 * chunck.shape[1] + c2]

            if len(chunck_noNaN) >= chunck.size / 2:
                if fittype == 2 and len(chunck_noNaN) >= 3:
                    m, _, rms = fit_bilinplane(chunck_noNaN, np.column_stack((c1, c2)))
                    medvalue = np.median(chunck_noNaN)
                    m = np.array([medvalue, 0, 0])
                elif fittype == 1:
                    medvalue = np.median(chunck_noNaN)
                    tmp = np.ones_like(chunck_noNaN) * medvalue
                    dif = chunck_noNaN - tmp
                    rms = np.sqrt(np.mean(dif ** 2))
                    m = np.array([medvalue, 0, 0])
                else:
                    meanvalue = np.mean(chunck_noNaN)
                    tmp = np.ones_like(chunck_noNaN) * meanvalue
                    dif = chunck_noNaN - tmp
                    rms = np.sqrt(np.mean(dif ** 2))
                    m = np.array([meanvalue, 0, 0])

                if rms <= tolerance:
                    oldindmat[k, icol - 3:icol] = np.array([1, *m])
                else:
                    oldindmat[k, icol - 3:icol] = np.array([0, *m])
            elif len(chunck_noNaN) < 1:
                oldindmat[k, icol - 3:icol] = np.array([1, np.nan, np.nan, np.nan])
            else:
                oldindmat[k, icol - 3:icol] = np.array([0, np.nan, np.nan, np.nan])

    newindmat = oldindmat.copy()
    return newindmat

def quadtree_part(data, tolerance, fittype, startlevel=1, maxdim=13):
    # Get size of data-file
    lin, col = data.shape

    # Adjust data size to be a power of 2
    dim = 1
    condition = max([lin, col])
    while condition > 2 ** dim:
        dim = dim + 1
    print(dim)
    
    nlin = 2 ** dim
    ncol = nlin

    dataexp = np.full((nlin, ncol), np.nan)
    dataexp[0:lin, 0:col] = data

    # Initialize the quadtree index matrix
    indmat = np.array([[1, 0, 10, 0, 0],
                       [2, 0, 10, 0, 0],
                       [3, 0, 10, 0, 0],
                       [4, 0, 10, 0, 0]])

    # Add levels to the index matrix if startlevel is greater than 1
    if startlevel > 1:
        for k in range(2, startlevel + 1):
            nindmat = quadtree_level(indmat)
            indmat = nindmat

    # Loop over each k in 2^k
    for k in range(startlevel, maxdim + 1):
        newindmat = check_quadtree(indmat, dataexp, tolerance, fittype)
        che = newindmat[:, -4]

        # If any zeros in check-column, perform further partitioning
        if np.prod(che) == 0:
            indmat = quadtree_level(newindmat)
        else:
            k = maxdim

    # Plot the quadtree points and squares
    cntp, co2, cx, cy = plot_quadtree(newindmat, dataexp)

    # Plot everything with patches
    sqval = newindmat[:, -3]
    plt.figure()
    plt.pcolormesh(cx, cy, sqval, shading='auto')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()

    return newindmat, sqval, cx, cy, cntp, nlin