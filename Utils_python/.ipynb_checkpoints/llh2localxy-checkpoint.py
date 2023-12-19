import numpy as np 
def polyconic(Lat, Diff_long, Lat_Orig):
    p1 = Lat_Orig
    p2 = Lat
    il = Diff_long

    arcone = 4.8481368e-6
    esq = 6.7686580e-3
    la = 6378206.4
    a0 = 6367399.7
    a2 = 32433.888
    a4 = 34.4187
    a6 = 0.0454
    a8 = 6.0e-5

    ip = p2 - p1
    sinp2 = np.sin(p2 * arcone)
    cosp2 = np.cos(p2 * arcone)
    theta = il * sinp2
    a = np.sqrt(1.0 - (esq * (2. * sinp2))) / (la * arcone)
    cot = cosp2 / sinp2
    x = (cot * np.sin(theta * arcone)) / (a * arcone)
    ipr = ip * arcone
    pr = ((p2 + p1) / 2.) * arcone
    y = ((((a0 * ipr) - ((a2 * np.cos(2. * pr)) * np.sin(ipr))) + \
          ((a4 * np.cos(4. * pr)) * np.sin(2. * ipr))) - \
          ((a6 * np.cos(6. * pr)) * np.sin(3. * ipr))) + \
          ((a8 * np.cos(8. * pr)) * np.sin(4. * ipr))

    xy = [x, y]
    return xy

def llh2localxy(llh, ll_org):
    # Convert from decimal degrees to decimal seconds
    '''
    llh = np.array([[34.05, 34.10, 34.15],  # Latitude values
                [-118.24, -118.20, -118.10]])  # Longitude values

    # Latitude of origin in decimal degrees
    ll_org = np.array([34.10, -118.20])

    # Call the llh2localxy function
    xy = llh2localxy(llh, ll_org)

    # Print the result
    print("Local XY Coordinates (in km):")
    print(xy)
    '''
    lat = 3600.0 * llh[0, :]
    lon = 3600.0 * llh[1, :]

    Lat_Orig = 3600.0 * ll_org[0]
    Diff_long = 3600.0 * ll_org[1] * np.ones(lon.shape) - lon

    nsta = len(lon)
    xy = np.zeros((nsta, 2))

    for i in range(nsta):
        xy[i, :] = polyconic(lat[i], Diff_long[i], Lat_Orig)

    # Convert units from meters to kilometers and flip the x-axis
    xy[:, 0] = -xy[:, 0] / 1000.0
    xy[:, 1] = xy[:, 1] / 1000.0

    return xy