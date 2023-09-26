from datetime import datetime as dt
import numpy as np

def datenum(d):
    '''
    Serial date number
    used for SBI_Year
    '''
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

def SBI_Year(imd):
    '''
    A simple script that takes in numbers on the format '19990930'
    and changes to a number format 1999.7590
    imd - numpy (n,1) Vector with dates in strings
    out - numpy (n,1) Vector with dates in int
    created by Rishabh Dutta
    '''
    dstr = imd
    nd = imd.shape[0]
    out = np.zeros((nd,1))
    for k in range(nd):
        # get first day of year, minus 1/2 a day:
        d1 = dt.strptime(dstr[k][0][0:4]+'0101', '%Y%m%d')
        dn1 = datenum(d1) - 0.5 
        # get last day of year, plus 0.5
        d2 = dt.strptime(dstr[k][0][0:4]+'1231', '%Y%m%d')
        dne = datenum(d2) + 0.5
        # get number of days in that year:
        ndays = dne - dn1 

        # get day of year:
        d3 = dt.strptime(dstr[k][0], '%Y%m%d')
        doy = datenum(d3) - dn1 
        # get fractional year:
        fracyr = doy/ndays
        out[k] = int(dstr[k][0][0:4])+ fracyr 
    return out