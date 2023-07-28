import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='logs.log',
                    encoding='utf-8',
                    format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)

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

clusterName = "M9"
clusterNameFile = ("{}.phot".format(clusterName))

dat = Table.read(clusterNameFile, format="ascii")

u = dat['col10']
b = dat['col4']
v = dat['col8']
i = dat['col6']
chi = dat['col12']
sharp = dat['col13']

ind = np.where(dat['col1'] == 320)[0]
cond = np.logical_or.reduce((b>60,v>60, chi>3, abs(sharp)>0.5))
    #cond = np.logical_and.reduce((b<60,v<60))
ind = np.where(cond)[0]

print(dat[ind])

dat1 = dat[ind]

moo = np.where(dat1['col1'] == 320)[0]
print(moo)

df2 = Table.read("clusterMembers/M9_memberStars_testing.dat", format="ascii", delimiter="\s")
df3 = Table.read("candStars/candStarMasterList/M9_candStarsMaster_testing.dat", format="ascii", delimiter="\s")
# print(df2)

vRaw = df2['col14']
B = df2['col13']
bvRaw = B-vRaw

v = df2['col10']
b = df2['col9']
i = df2['col11']
bv = b-v
vi = v-i

# print(df2)

# if (vi[1543] > 0.331+1.444*bv[1543]):
#     print(0)
#     logging.error('Run unsuccessful')
# else:
#     print(1)
#     logging.info('Run successful')



vRaw1 = df3['col14']
B1 = df3['col13']
bvRaw1 = B1-vRaw1

v1 = df3['col11']
b1 = df3['col10']
bv1 = b1-v1

def model_f(x,a,b,c,d,e,f,g,k):
    x=x+k
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g

# fig, ax = plt.subplots()
# ax.scatter(bvRaw, vRaw, c='k', s=0.1)
# # ax.scatter(bvhb,vhb,c='b',s=2)
# ax.scatter(bvRaw1, vRaw1, c='orangered', s=5, marker="o")
# xplot = np.linspace(bv1.min(), bv1.max(), len(bv1))
# # -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15
# y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35,-0.15)
# # ax.plot(bv1, y, color="red", linestyle="--")
# # ax.set_title("{} $E(B-V)$={:.2f} $m-M$={:.2f}".format(clusterName, ebv, distModulus))
# ax.set_xlim(-0.75, 1.3)
# ax.set_ylim(22,12)
# ax.set_xlabel('$B-V$')
# ax.set_ylabel('$V$')
# plt.show()

