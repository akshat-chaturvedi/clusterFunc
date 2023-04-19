import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from clusterDicts import *
import glob

bhArray = []
starCount = []
magArray = []

for name in glob.glob('candStars/TSVs/uBVCandStars/*_candStars_6.dat'):
    with open(r"%s"%name, 'r') as fp:
        lines = len(fp.readlines()) - 2
        starCount += [lines]
        fileName = name.split('/')[3]
        clusterName = fileName.split('_')[0]
        bhArray += [locals()[clusterName]['bh']]
        magArray += [locals()[clusterName]['mv']]

starCount = np.array(starCount)
magArray = np.array(magArray)

yArray = starCount/(10**((4.83 - magArray)/2.5))

fig, ax = plt.subplots()
ax.scatter(bhArray, starCount, s=10, marker="*")
ax.set_xlim(-1.2,1.2)
ax.set_xlabel("B/H Parameter")
ax.set_ylabel("Number of blue stars")
ax.set_title("Number of stars as a function of HB morphology")
fig.savefig("starBH.jpg", dpi=300)

fig, ax = plt.subplots()
ax.scatter(bhArray, yArray, s=10, marker="*")
ax.set_xlim(-1.2,1.2)
ax.set_xlabel("B/H Parameter")
ax.set_ylabel("Normalized number of blue stars")
ax.set_title("Normalized number of stars as a function of HB morphology")
fig.savefig("starNormBH.jpg", dpi=300)
