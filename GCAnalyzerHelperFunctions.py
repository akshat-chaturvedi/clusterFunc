# Helper functions for GCAnalyzer
import numpy as np


def coordTransfer(ra, dec):
    raString = ra.split(" ")
    raHH = int(raString[0])
    raMM = int(raString[1])
    raSS = float(raString[2])

    decString = dec.split(" ")
    decDeg = int(decString[0])
    decMin = int(decString[1])
    decSec = float(decString[2])

    raDeg = raHH * 15 + raMM / 60 * 15 + raSS / 3600 * 15
    if decDeg < 0:
        decDeg = decDeg - decMin / 60 - decSec / 3600
    else:
        decDeg = decDeg + decMin / 60 + decSec / 3600

    return raDeg, decDeg

def degToRad(theta):
    radian = np.deg2rad(theta)
    return radian

def arcMinAngularSep(a1, d1, a2, d2):
    a1Rad = np.deg2rad(a1)
    a2Rad = np.deg2rad(a2)
    d1Rad = np.deg2rad(d1)
    d2Rad = np.deg2rad(d2)
    sep = np.sin(d1Rad)*np.sin(d2Rad)+np.cos(d1Rad)*np.cos(d2Rad)*np.cos(a2Rad-a1Rad)
    angularSep = np.arccos(sep)*206265/60
    return angularSep

def parallaxError(dist_kpc, parStar, parStarErr):
    parCluster = 1/dist_kpc
    # parStar = dat5["parallax"]
    # parStarErr = dat5["parallax_error"]
    parDiff = abs(parCluster-parStar)
    return parStarErr, parDiff

def tidalRadius(c, r_c):
    tidalRad = r_c*(10**c)
    return tidalRad

def HB_model(x, a, b, c, d, e, f, g, k):
    # TODO: add comments describing what changing each of the parameters here actually does vis-a-vis moving the HB
    x = x+k
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
