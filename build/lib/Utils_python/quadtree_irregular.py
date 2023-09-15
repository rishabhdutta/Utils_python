import numpy as np
from scipy.stats import nanvar
import matplotlib.pyplot as plt


def inbox(box, GPScoord):
    xcoord = GPScoord[:, 0]
    ycoord = GPScoord[:, 1]
    xmin, xmax, ymin, ymax = box

    indices = np.where(
        (xcoord <= xmax) & (xcoord > xmin) &
        (ycoord <= ymax) & (ycoord > ymin)
    )[0]

    return indices


def divideandvar(box, GPScoord, GPSmag):
    xhalf = np.mean([box[0], box[1]])
    yhalf = np.mean([box[2], box[3]])

    boxes = {}
    varsval = {}

    # box 1
    box1 = [box[0], xhalf, box[2], yhalf]
    inbox1 = inbox(box1, GPScoord)
    if len(inbox1) > 0:
        inboxval1 = GPSmag[inbox1]
        vars1 = np.nanvar(inboxval1)
    else:
        vars1 = np.nan

    # box 2
    box2 = [xhalf, box[1], box[2], yhalf]
    inbox2 = inbox(box2, GPScoord)
    if len(inbox2) > 0:
        inboxval2 = GPSmag[inbox2]
        vars2 = np.nanvar(inboxval2)
    else:
        vars2 = np.nan

    # box 3
    box3 = [box[0], xhalf, yhalf, box[3]]
    inbox3 = inbox(box3, GPScoord)
    if len(inbox3) > 0:
        inboxval3 = GPSmag[inbox3]
        vars3 = np.nanvar(inboxval3)
    else:
        vars3 = np.nan

    # box 4
    box4 = [xhalf, box[1], yhalf, box[3]]
    inbox4 = inbox(box4, GPScoord)
    if len(inbox4) > 0:
        inboxval4 = GPSmag[inbox4]
        vars4 = np.nanvar(inboxval4)
    else:
        vars4 = np.nan

    if not np.isnan(vars1):
        boxes['box1'] = box1
        varsval['var1'] = vars1

        if not np.isnan(vars2):
            boxes['box2'] = box2
            varsval['var2'] = vars2

            if not np.isnan(vars3):
                boxes['box3'] = box3
                varsval['var3'] = vars3

                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
            else:
                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
        else:
            if not np.isnan(vars3):
                boxes['box3'] = box3
                varsval['var3'] = vars3

                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
            else:
                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
    else:
        if not np.isnan(vars2):
            boxes['box2'] = box2
            varsval['var2'] = vars2

            if not np.isnan(vars3):
                boxes['box3'] = box3
                varsval['var3'] = vars3

                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
            else:
                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
        else:
            if not np.isnan(vars3):
                boxes['box3'] = box3
                varsval['var3'] = vars3

                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4
            else:
                if not np.isnan(vars4):
                    boxes['box4'] = box4
                    varsval['var4'] = vars4

    return boxes, varsval

def checkbox(box, GPScoord, GPSmag, limitvar):
    boxes, vars = divideandvar(box, GPScoord, GPSmag)
    a = list(boxes.values())
    num = len(a)
    var_a = list(vars.values())
    ind = []
    boxout = []

    for i in range(num):
        varbox = var_a[i]
        if varbox > limitvar:
            ind.append(i)
        else:
            boxno = list(boxes.keys())[i]
            boxout.append(boxno)

    boxmore = []
    if not not ind:
        for i in range(len(ind)):
            indd = ind[i]
            boxmore.extend(list(boxes.values())[indd])
    else:
        boxmore = []

    return boxout, boxmore

def qt_irregular(box, GPScoord, GPSmag, limitvar, plot1):
    boxout = []
    boxmore = box.copy()

    while len(boxmore) > 0:
        box1 = boxmore[0]
        boxmore = boxmore[1:]
        bxout, bxmore = checkbox(box1, GPScoord, GPSmag, limitvar)
        boxout.extend(bxout)
        boxmore.extend(bxmore)

    if plot1 == 1:
        plt.figure()
        for ind in boxout:
            x = [ind[0], ind[1], ind[1], ind[0], ind[0]]
            y = [ind[2], ind[2], ind[3], ind[3], ind[2]]
            plt.plot(x, y, 'k-')
        plt.scatter(GPScoord[:, 0], GPScoord[:, 1], s=100, c=GPSmag, marker='.')
        plt.axis(box)

    plt.show()

    return boxout




