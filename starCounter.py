import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from clusterDicts import *
import glob

bhArray = []
starCount = []
for name in glob.glob('candStars/TSVs/uBVCandStars/*_candStars_6.dat'):
    with open(r"%s"%name, 'r') as fp:
        lines = len(fp.readlines()) - 2
        starCount += [lines]
        fileName = name.split('/')[3]
        clusterName = fileName.split('_')[0]
        bhArray += [locals()[clusterName]['bh']]

plt.scatter(bhArray, starCount, s=10, marker="*")
plt.xlim(-1.2,1.2)
plt.xlabel("B/H Parameter")
plt.ylabel("Number of blue stars")
plt.title("Number of stars as a function of HB morphology")
plt.savefig("starBH.jpg", dpi=300)

