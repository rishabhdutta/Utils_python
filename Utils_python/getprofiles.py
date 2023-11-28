import numpy as np 
from Utils_python import inpoly

def get_profiles(x1d, y1d, mat, p1xy, p2xy, wid): 
    '''
    inputs---
    mat - 2D matrix of data
    x1d - lon coords of data in 1D
    y1d - lat coords of data in 1D
    p1xy, p2xy - lon and lat coords of 2 points of profile
    wid - width of profile in coords
    
    outputs---
    prof_x, prof_y, diff_y 
    '''
    
    # first get the 2d matrices 
    x2d, y2d = np.meshgrid(x1d,y1d)
    
    # get the slope and yint with the two points 
    m1 = (p2xy[1] - p1xy[1])/(p2xy[0] - p1xy[0])  # m = (y2-y1)/(x2-x1)
    yint1 = p2xy[1] - m1 *p2xy[0]
    
    m2 = -1/m1 # perp slope
    yint2 = p1xy[1] - m2* p1xy[0]  # y-int for ref p1xy
    yint3 = p2xy[1] - m2* p2xy[0]  # y-int for ref p2xy
    
    # get the 2 corner points with ref p1xy
    r1x = p1xy[0] + wid/(np.sqrt(1+m2**2))
    r2x = p1xy[0] - wid/(np.sqrt(1+m2**2))
    
    r1y = m2*r1x + yint2
    r2y = m2*r2x + yint2 
    
    # get the 2 corner points with ref p2xy
    r3x = p2xy[0] + wid/(np.sqrt(1+m2**2))
    r4x = p2xy[0] - wid/(np.sqrt(1+m2**2))
    
    r3y = m2*r3x + yint3
    r4y = m2*r4x + yint3
    
    # circular way for corners - r1 r2 r4 r3 
    box_x = np.array([r1x,r2x,r4x,r3x])
    box_y = np.array([r1y,r2y,r4y,r3y])
    
    inbox = inpoly.inpolygon(x2d, y2d, box_x, box_y)
    xboxall = x2d[inbox == 1]
    yboxall = y2d[inbox == 1]
    
    matall = mat[inbox == 1]
    
    # loop over all the points 
    # ref line is y = m2 * x + yint2 or m2*x - y + yint2 = 0
    prof_x = np.zeros((xboxall.shape[0]))
    prof_y = np.zeros((xboxall.shape[0]))
    for i in range(xboxall.shape[0]):
        xboxval = xboxall[i]
        yboxval = yboxall[i]
        
        distfromline = np.abs(m2*xboxval - yboxval + yint2)/(np.sqrt(m2**2 + 1))
        
        prof_x[i] = distfromline
        prof_y[i] = matall[i]
        
    nan_indices = np.isnan(prof_y)
    prof_y_cleaned = prof_y[~nan_indices]
    prof_x_cleaned = prof_x[~nan_indices]
    
    coefficients = np.polyfit(prof_x_cleaned, prof_y_cleaned, 1)
    # Create a polynomial function using the coefficients
    poly_function = np.poly1d(coefficients)
    
    y_fit = poly_function(prof_x)
    diff_y = prof_y - y_fit
        
    return prof_x, prof_y, diff_y