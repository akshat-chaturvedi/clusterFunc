import numpy as np
import pandas as pd
from astropy.table import Table

df = pd.read_csv("clusterMembers/M79_memberStars_6.dat", delimiter="\t", skiprows=2)

df1 = pd.read_csv("nonMembers/M79_nonMembers.dat", delimiter="\t", skiprows=2)

starNums= df.iloc[:,0]

starNums1 = df1.iloc[:,0]

#print(starNums[0])

lst = np.array([3524,3473,3491,3487,3353,2600,2363,1995, 816])

# print(lst)

starNums = np.array([starNums])

starNums1 = np.array([starNums1])

# print(starNums)

a = np.intersect1d(starNums, lst)

c = np.setdiff1d(lst, starNums)

# print(a)
# print(c)

b = np.intersect1d(starNums1, lst)

# print(b)

clusterName = "M80"
clusterNameFile = ("{}.phot".format(clusterName))

dat = Table.read(clusterNameFile, format="ascii")

u = dat['col10']
b = dat['col4']
v = dat['col8']
i = dat['col6']
chi = dat['col12']
sharp = dat['col13']

ind = np.where(dat['col1'] == 6422)[0]
cond = np.logical_or.reduce((b>60,v>60, chi>3, abs(sharp)>0.5))
    #cond = np.logical_and.reduce((b<60,v<60))
ind = np.where(cond)[0]

# print(dat[ind])

dat1 = dat[ind]

moo = np.where(dat1["col1"]==10002)[0]

print(dat1[moo])
