import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
from astropy.table import Table
from astropy.table import Column
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
import pandas as pd
from collections import Counter
import seaborn as sns
from PyAstronomy import pyasl

clusterName = input("Enter a cluster name:")
clusterNameFile = ("{}.phot".format(clusterName))
clusterGaiaFile = ("NewCoor/{}_new_coor.dat".format(clusterName))
clusterGaiaResults = ("{}.csv".format(clusterName))
dist = float(input("Enter a distance:"))
ebv = float(input("Enter a E(B-V) value:"))
raClust = str(input("Enter the RA with spaces:"))
decClust = str(input("Enter the Dec with spaces:"))
c = float(input("Enter King-model central concentration value:"))
r_c = float(input("Enter a core radius in arcminutes:"))

dat = Table.read(clusterNameFile, format="ascii")
dat2 = Table.read(clusterGaiaFile, format="ascii")
#print(dat2)
dat4 = pd.read_csv("GAIAData/{}".format(clusterGaiaResults))
dat4 = dat4.fillna(1000)
dat5 = dat4[["pmra","pmdec"]].copy()
#print(dat4)
'''
clusters = DBSCAN(eps=0.7, min_samples=3).fit(dat5)
plt.figure(figsize=(8,6))
p = sns.scatterplot(data=dat5, x="pmra", y="pmdec", hue=clusters.labels_, legend="brief", palette="deep")
sns.move_legend(p, "upper center", bbox_to_anchor=(0.15,0.42), fontsize = "x-small", ncol=3, title='Clusters')
plt.title("Proper Motion Clustering")
plt.show()
'''
#print(dat)

def coordTransfer(ra, dec):
    raString = ra.split(" ")
    raHH = int(raString[0])
    raMM = int(raString[1])
    raSS = float(raString[2])

    decString = dec.split(" ")
    decDeg = int(decString[0])
    decMin = int(decString[1])
    decSec = float(decString[2])

    raDeg = raHH*15+raMM/60*15+raSS/3600*15
    if decDeg < 0:
        decDeg = decDeg-decMin/60-decSec/3600
    else:
        decDeg = decDeg+decMin/60+decSec/3600

    return raDeg, decDeg

def degToRad(theta):
    radian = np.deg2rad(theta)
    return radian

def tidalRadius(c, r_c):
    tidalRad = r_c*(10**c)
    return tidalRad

def arcMinAngularSep(a1, d1, a2, d2):
    a1Rad = np.deg2rad(a1)
    a2Rad = np.deg2rad(a2)
    d1Rad = np.deg2rad(d1)
    d2Rad = np.deg2rad(d2)
    sep = np.sin(d1Rad)*np.sin(d2Rad)+np.cos(d1Rad)*np.cos(d2Rad)*np.cos(a2Rad-a1Rad)
    angularSep = np.arccos(sep)*206265/60
    return angularSep

def parallaxError(dist_kpc):
    parCluster = 1/(dist_kpc*1000)
    parStar = dat4["parallax"]
    parStarErr = dat4["parallax_error"]
    parDiff = abs(parCluster-parStar)
    return parStarErr, parDiff

def dereddening(uinit,binit,vinit,iinit,ebv,ru,rb,rv,ri,dist_kpc):
    bv = binit-vinit
    u = uinit - 5*np.log10(dist_kpc*100) - (ru * ebv - 0.00341 * bv - 0.0131 * ebv * bv)
    b = binit - 5*np.log10(dist_kpc*100) - (rb * ebv - 0.0454 * bv - 0.142 * ebv * bv)
    v = vinit - 5*np.log10(dist_kpc*100) - (rv * ebv - 0.0143 * bv - 0.0568 * ebv * bv)
    i = iinit - 5*np.log10(dist_kpc*100) - ri*ebv
    return u, b, v, i        

def gethbtop(bv,v):
    cond = v-bv>5
    return bv[cond], v[cond]

#Plots that need fitting

def DBSCANPlots(bv,v,b,u,vi,ub,ubbv,clusterName,cond,dist,raClust,decClust,c,r_c, uRaw, vRaw, bRaw,ebv):
    clusters = DBSCAN(eps=0.3, min_samples=5).fit(dat5)
    labels = DBSCAN(eps=0.7, min_samples=3).fit_predict(dat5)
    plt.figure(figsize=(8,6))
    p = sns.scatterplot(data=dat5, x="pmra", y="pmdec", hue=clusters.labels_, legend="brief", palette="deep")
    sns.move_legend(p, "upper center", bbox_to_anchor=(0.15,0.42), fontsize = "x-small", ncol=3, title='Clusters')
    plt.title("Proper Motion Clustering")
    plt.xlim(-50,50)
    plt.ylim(-50,50)
    plt.savefig("DBSCANPlots/%s.png"%(clusterName), bbox_inches="tight",dpi=300)
    #print(dat4.iloc[labels==0,2])
    ind = dat4[f"{clusterName.lower()}_oid"]-1
    indLab = np.where(labels==0)[0]
    indSiegel = ind[indLab]
    indAll = np.intersect1d(indSiegel, cond)

    clusterCore = coordTransfer(raClust, decClust)
    angSep = arcMinAngularSep(clusterCore[0],clusterCore[1], dat2["col2"],dat2["col3"])
    parStarErr, parDiff = parallaxError(dist)
    parCond = np.where(parDiff < 3*parStarErr)[0]
    tidalRad = tidalRadius(c,r_c)
    tidalCond = np.where(angSep<tidalRad)[0]
    indAll = np.intersect1d(indAll, tidalCond)
    indParSiegel = ind[parCond]
    indAll = np.intersect1d(indAll, indParSiegel)
    
    #breakpoint()
    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],v[indAll],c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv[indAll].min(),bv[indAll].max(),len(bv[indAll]))
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -4, ymax = 5)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    UVBrightCond = np.logical_and(v[indAll]<y2,bv[indAll]<-0.05)

    UVBrightRA = dat2["col2"][indAll][UVBrightCond]
    UVBrightDec = dat2["col3"][indAll][UVBrightCond]

    coordArrayRA = []
    coordArrayDec = []
    for i in range(0,len(UVBrightRA)):
        sexa = pyasl.coordsDegToSexa(UVBrightRA[i],UVBrightDec[i])
        coordArrayRA.append(sexa[0:12])
        coordArrayDec.append(sexa[14:])
        
    #fileArray = np.column_stack((indAll[UVBrightCond],dat2["col2"][indAll][UVBrightCond],dat2["col3"][indAll][UVBrightCond],bv[indAll][UVBrightCond],v[indAll][UVBrightCond]))
    #fileArray1 = np.column_stack((indAll[UVBrightCond],coordArrayRA,coordArrayDec,bv[indAll][UVBrightCond],v[indAll][UVBrightCond]))
    #breakpoint()
    #infiles = np.savetxt("candStars/%s_candStars.dat"%(clusterName),fileArray, fmt="%d,%.6f,%.6f,%.2f,%.2f", delimiter = ",",header="SiegelIndex,RA,Dec,(B-V)0,Mv")
    #infiles1 = np.savetxt("candStars/%s_candStars1.dat"%(clusterName),fileArray1, fmt="%s", delimiter = "\t",header="Siegel_Index    RA    Dec    (B-V)_0    M_V")
    
    ax.scatter(bv[indAll][UVBrightCond],v[indAll][UVBrightCond],c='g',s=2)
    ax.set_xlim(-0.75,1.6)
    ax.set_ylim(5,-4)
    ax.set_xlabel('(B$-$V)$_0$', style="italic")
    ax.set_ylabel('M$_V$', style="italic")
    #fig.savefig('Plots/DBSCAN/VBV/VvsBV_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)
    fig.savefig('Plots/DBSCAN/VBV/VvsBV_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)    
    #breakpoint()
    
    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],b[indAll],c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot-0.2, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    #y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    #UVBrightCond = np.logical_and(b[indAll]<y2,bv[indAll]<-0.05)
    ax.scatter(bv[indAll][UVBrightCond],b[indAll][UVBrightCond],c='g',s=2)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-1.3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('B')
    fig.savefig('Plots/DBSCAN/BBV/BvsBV_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)
    
    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],u[indAll],c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1)
    #ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    #y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1)
    #UVBrightCond = np.logical_and(u[indAll]<y2,bv[indAll]<-0.05)
    ax.scatter(bv[indAll][UVBrightCond],u[indAll][UVBrightCond],c='g',s=2)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(5,-4)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U')
    fig.savefig('Plots/DBSCAN/UBV/UvsBV_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)


    fileArray = np.column_stack((indAll[UVBrightCond],dat2["col2"][indAll][UVBrightCond],dat2["col3"][indAll][UVBrightCond],bv[indAll][UVBrightCond],u[indAll][UVBrightCond],b[indAll][UVBrightCond],v[indAll][UVBrightCond],uRaw[indAll][UVBrightCond],bRaw[indAll][UVBrightCond],vRaw[indAll][UVBrightCond]))
    fileArray1 = np.column_stack((indAll[UVBrightCond],coordArrayRA,coordArrayDec,bv[indAll][UVBrightCond],v[indAll][UVBrightCond]))
    #breakpoint()
    infiles = np.savetxt("candStars/%s_candStars.dat"%(clusterName),fileArray, fmt="%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f", delimiter = ",",header="SiegelIndex,RA,Dec,(B-V)0,Mu,Mb,Mu,u,b,v,E(B-V)={}".format(ebv))
    infiles1 = np.savetxt("candStars/%s_candStars1.dat"%(clusterName),fileArray1, fmt="%s", delimiter = "\t",header="Siegel_Index    RA    Dec    (B-V)_0    M_V")

    
    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],ub[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/DBSCAN/UBBV/UBvsBV_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)
    
    fig, ax = plt.subplots()
    ax.scatter(vi[indAll],ub[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/DBSCAN/UBVI/UBvsVI_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)
    
    fig, ax = plt.subplots()
    ax.scatter(vi[indAll],ubbv[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('(U-B)-(B-V)')
    fig.savefig('Plots/DBSCAN/UBBVVI/UBBVvsVI_%s_DBSCAN.png'%(clusterName),bbox_inches='tight',dpi=300)

def VBVplotBestFit(bv,v,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,v,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('V')
    fig.savefig('Plots/VBV/VvsBV_%s.png'%(clusterName),bbox_inches='tight',dpi=300)

def BBVplotBestFit(bv,b,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,b,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot-0.2, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('B')
    fig.savefig('Plots/BBV/BvsBV_%s.png'%(clusterName),bbox_inches='tight',dpi=300)


def UBVplotBestFit(bv,u,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,u,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U')
    fig.savefig('Plots/UBV/UvsBV_%s.png'%(clusterName),bbox_inches='tight',dpi=300)

#Plots that don't need fitting

def UBBVplotBestFit(bv,ub,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,ub,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/UBBV/UBvsBV_%s.png'%(clusterName),bbox_inches='tight',dpi=300)

def UBVIplotBestFit(vi,ub,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(vi,ub,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/UBVI/UBvsVI_%s.png'%(clusterName),bbox_inches='tight',dpi=300)

def UBBVVIplotBestFit(vi, ubbv, clusterName):
    fig, ax = plt.subplots()
    ax.scatter(vi,ubbv,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('(U-B)-(B-V)')
    fig.savefig('Plots/UBBVVI/UBBVvsVI_%s.png'%(clusterName),bbox_inches='tight',dpi=300)

def model_f(x,a,b,c,d,e,f,g):
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g

def clusterFunc(dat,ebv,dist,rv=3.164,ru=4.985,rb=4.170, ri = 1.940):
    # Start with raw cluster data
    u = dat['col10']
    b = dat['col4']
    v = dat['col8']
    i = dat['col6']
    chi = dat['col12']
    sharp = dat['col13']
    ra = dat2['col2']
    dec = dat2['col3']
    # Remove missing magnitudes and bad fits (chi^2 and sharp features also)
    #breakpoint()
    #cond = np.logical_and.reduce((u<90,b<90,v<90, chi<3, abs(sharp)<0.5))
    cond = np.logical_and.reduce((b<90,v<90))
    ind = np.where(cond)[0]
    u1, b1, v1, i1 = u[cond], b[cond], v[cond], i[cond]
    bv1 = b1-v1
    ra1, dec1 = ra[cond], dec[cond]
    # De-redden the cluster
    #u2, b2, v2, i2 = dereddening(u1, b1, v1, i1, ebv, ru, rb, rv, ri, dist)
    u2, b2, v2, i2 = dereddening(u, b, v, i, ebv, ru, rb, rv, ri, dist)
    bv = b2-v2 # Define a color
    vi = v2-i2
    ub = u2-b2
    ubbv = (u2-b2)-(b2-v2)
    g = v - 0.0124*(b-v)
    dat1 = Table((g, ra, dec), names=("g","ra","dec"))
    #print(dat1[:][4])
    #dat.add_column(g, name="g_mag")
    #dat1.write("gData/%s.csv"%(clusterName), format="ascii.csv", overwrite=True)
    '''
    plt.figure()
    plt.scatter(bv,v2, c='k',s=0.1)
    x_array = np.linspace(-0.6,1.3,100)
    y = model_f(x_array, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2)
    plt.plot(x_array, y, "r-")
    plt.ylim(4,-2)
    plt.xlim(-0.6,1.3)
    plt.show()
    '''
    # Select just the points at the top of the HB
    #bvhb, vhb = gethbtop(bv,v2)
    #breakpoint()
    # Fit the curve of your choosing (in this case, f)
    #popt, pcov = curve_fit(model_f,bvhb,vhb,p0=[-3.74,7.03,6.83,-19.86,8.98,-1.51,0.20])
    # Plot everything!
    VBVplotBestFit(bv,v2,clusterName)
    UBVplotBestFit(bv,u2,clusterName)
    UBBVplotBestFit(bv,ub,clusterName)
    UBVIplotBestFit(vi,ub,clusterName)
    UBBVVIplotBestFit(vi,ubbv,clusterName)
    BBVplotBestFit(bv,b2,clusterName)
    DBSCANPlots(bv,v2,b2,u2,vi,ub,ubbv,clusterName, ind,dist,raClust,decClust,c,r_c,u,v,b,ebv)
    # You're done for this cluster!
    #print(dat)

def main():
    #ebv_list = np.random.rand()
    #dists = 10*np.random.rand()
    clusterFunc(dat, ebv, dist, rv=3.315,ru=5.231,rb=4.315, ri = 1.940)

if __name__ == "__main__":
    main()
    
