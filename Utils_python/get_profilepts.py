import numpy as np

def getprofilepoints(clicked_points, widval):
    xval1, yval1 = clicked_points[0]
    xval2, yval2 = clicked_points[1]

    # use the first point
    xcen, ycen = xval1, yval1

    # get the slope and y-intercept
    mval = (yval2 - yval1) / (xval2 - xval1)
    mval2 = -1 / mval
    yint2 = ycen - mval2 * xcen

    # get points
    xprof1 = xcen - widval / np.sqrt(1 + mval2**2)
    xprof2 = xcen + widval / np.sqrt(1 + mval2**2)

    yprof1 = mval2 * xprof1 + yint2
    yprof2 = mval2 * xprof2 + yint2

    pairval = np.array([[xprof1, yprof1], [xprof2, yprof2]])

    return pairval