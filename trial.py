import numpy as np 
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.cluster import DBSCAN
import pandas as pd
import seaborn as sns
from PyAstronomy import pyasl
from clusterDicts import *
import time
import logging

#Last updated July 5 2023
## Explainer: Currently creating CMDs using unprocessed (dereddening), observed mag values. CandStar lists of both UBV
## and VBV are produced. HEB checking individuals.

ext = "5Sigma"

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
    #tidalRad = r_c*(np.exp(c))
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
    parCluster = 1/(dist_kpc)
    parStar = dat5["parallax"]
    parStarErr = dat5["parallax_error"]
    parDiff = abs(parCluster-parStar)
    return parStarErr, parDiff

def dereddening(uinit,binit,vinit,iinit,ebv,ru,rb,rv,ri,dist_kpc):
    bv = binit-vinit
    u = uinit - 5*np.log10(dist_kpc*100) - (ru * ebv - 0.00341 * bv - 0.0131 * ebv * bv)
    b = binit - 5*np.log10(dist_kpc*100) - (rb * ebv - 0.0454 * bv - 0.142 * ebv * bv)
    v = vinit - 5*np.log10(dist_kpc*100) - (rv * ebv - 0.0143 * bv - 0.0568 * ebv * bv)
    i = iinit - 5*np.log10(dist_kpc*100) - ri*ebv
    return u, b, v, i        

def dereddening1(binit,vinit,ebv,dist_kpc):
    b = binit - 5*np.log10(dist_kpc*100)
    v = vinit - 5*np.log10(dist_kpc*100)
    return b,v

def gethbtop(bv,v):
    cond = v-bv>5
    return bv[cond], v[cond]

#Plots that need fitting

def DBSCANPlots(bv,v,b,u,vi,ub,ubbv,clusterName,cond,dist,raClust,decClust,c,r_c, uRaw, vRaw, bRaw, iRaw, ebv, flagArray):
    dat6 = pd.concat([dat5["pmra"],dat5["pmdec"]], axis="columns")
    fig, ax = plt.subplots()
    ax.scatter(dat6['pmra'], dat6['pmdec'], s = 0.1, marker="x")
    ax.set_title("Proper Motions")
    ax.set_xlabel("pmra [mas yr$^{-1}$]")
    ax.set_ylabel("pmdec [mas yr$^{-1}$]")
    ax.set_xlim(-20, 15)
    ax.set_ylim(-15, 15)
    fig.savefig("PMPlots/%s_%s.png" % (clusterName, ext), bbox_inches="tight", dpi=300)

    clusters = DBSCAN(eps=0.3, min_samples=5).fit(dat6)
    #clusters = DBSCAN(eps=0.9, min_samples=2).fit(dat5)
    labels = DBSCAN(eps=0.7, min_samples=3).fit_predict(dat6)
    #labels = DBSCAN(eps=0.9, min_samples=2).fit_predict(dat5)
    plt.figure(figsize=(8,6))
    #colors = sns.color_palette("husl",3)
    p = sns.scatterplot(data=dat5, x="pmra", y="pmdec",size=0.01,marker="+" ,hue=clusters.labels_, legend="brief", palette="deep")
    #breakpoint()
    sns.move_legend(p, "upper center", bbox_to_anchor=(0.15,0.42), fontsize = "x-small", ncol=3, title='Clusters')
    plt.title("Proper Motion Clustering")
    #breakpoint()
    plt.xlabel("pmra [mas yr$^{-1}$]")
    plt.ylabel("pmdec [mas yr$^{-1}$]")
    plt.xlim(-20,15)
    plt.ylim(-15,15)
    plt.savefig("DBSCANPlots/%s_%s.png"%(clusterName,ext), bbox_inches="tight",dpi=300)
    #print(dat4.iloc[labels==0,2])
    #indSiegel = dat5[f"{clusterName.lower()}_oid"]
    indSiegel = dat5[f"{clusterName.lower()}_oid"]-1
    indSiegel = np.asarray(indSiegel)
    indLab = np.where(labels==0)[0]
    ## Plotting for stars not in cluster
    # notInClust = np.where(labels!=0)[0]
    # notInClustSiegel = indSiegel[notInClust]
    ########################################
    #indLab = np.arange(len(labels))
    indSiegelDB = indSiegel[indLab]
    indAll = np.intersect1d(indSiegelDB, cond)
##    breakpoint()
    #indAll = np.arange(len(bv))

    clusterCore = coordTransfer(raClust, decClust)
    angSep = arcMinAngularSep(clusterCore[0],clusterCore[1], dat2["col2"],dat2["col3"])
    parStarErr, parDiff = parallaxError(dist)
    parCond = np.where(parDiff < 5*parStarErr)[0]
    tidalRad = tidalRadius(c,r_c)
    tidalCond = np.where(angSep<tidalRad)[0]
    indAll = np.intersect1d(indAll, tidalCond)
    #indParSiegel = ind[parCond]
    indParSiegel = indSiegel[parCond]
    indAll = np.intersect1d(indAll, indParSiegel)

    ## Plotting for stars not in cluster
    notInTidalRadCond = np.where(angSep>tidalRad)[0]
    notInParCond = np.where(parDiff > 3*parStarErr)[0]
    notInParCond = indSiegel[notInParCond]
    notInClust = np.where(labels != 0)[0]

    nonMemberCond = np.concatenate((notInTidalRadCond, notInParCond, notInClust))
    nonMemberCond = np.unique(nonMemberCond)

    notInClustSiegel = np.intersect1d(indSiegel, nonMemberCond)
    ########################################


    distModulus = 5*np.log10(dist*100)

    #HB Model Equation
    y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35,0.05)
    y_u = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 1.65, 0.8)

    #y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    UVBrightCond = np.logical_and(v[indAll]<y2,bv[indAll]<-0.05)

    ############RAW STAR MAG CMDS######################

    bvRaw = bRaw - vRaw


    fig, ax = plt.subplots()
    ax.scatter(bvRaw[indAll], vRaw[indAll], c='k', s=0.1)
    # ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bvRaw[indAll].min(), bvRaw[indAll].max(), len(bvRaw[indAll]))
    # x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    #ax.plot(xplot, y, "r-")
    ax.vlines(x=-0.05, ymin=-4, ymax=5)
    # ax.plot(xplot,model_f(xplot,*popt),'r--')
    # y2 = model_f(bvRaw[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    # y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    ax.scatter(bvRaw[indAll][UVBrightCond], vRaw[indAll][UVBrightCond], c='g', s=2)
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName, ebv, distModulus))
    ax.set_xlim(-0.75, 1.3)
    ax.set_ylim(21, 11.5)
    ax.set_xlabel('$B-V$')
    ax.set_ylabel('$V$')
    fig.savefig('Plots/DBSCAN/RAW/VBV/VvsBV_%s_DBSCAN_RAW_%s.png'%(clusterName,ext), bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots()
    ax.scatter(bvRaw[indAll], uRaw[indAll], c='k', s=0.1)
    # ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bvRaw[indAll].min(), bvRaw[indAll].max(), len(bvRaw[indAll]))
    # x_array = np.linspace(-0.6,1.3,100)
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    UVBrightCond1 = np.logical_and(u[indAll] < y_u, bv[indAll] < -0.05)
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    #ax.plot(xplot, y, "r-")
    ax.vlines(x=-0.05, ymin=-4, ymax=5)
    # ax.plot(xplot,model_f(xplot,*popt),'r--')
    # y2 = model_f(bvRaw[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    # y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    ax.scatter(bvRaw[indAll][UVBrightCond1], uRaw[indAll][UVBrightCond1], c='g', s=2)
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName, ebv, distModulus))
    ax.set_xlim(-0.75, 1.3)
    ax.set_ylim(21, 13.5)
    ax.set_xlabel('$B-V$')
    ax.set_ylabel('$u$')
    fig.savefig('Plots/DBSCAN/RAW/UBV/UvsBV_%s_DBSCAN_RAW_%s.png' %(clusterName, ext), bbox_inches='tight', dpi=300)

    ############RAW STAR MAG CMDS######################

    UVBrightRA = dat2["col2"][indAll][UVBrightCond]
    UVBrightDec = dat2["col3"][indAll][UVBrightCond]

    UVBrightRA1 = dat2["col2"][indAll]
    UVBrightDec1 = dat2["col3"][indAll]

    coordArrayRA = []
    coordArrayDec = []
    coordArrayRA1 = []
    coordArrayDec1 = []

    for i in range(0,len(UVBrightRA)):
        sexa = pyasl.coordsDegToSexa(UVBrightRA[i],UVBrightDec[i])
        sexaLen = len(sexa)-1
        coordArrayRA.append(sexa[0:12])
        coordArrayDec.append(sexa[14:sexaLen])

    for i in range(0,len(UVBrightRA1)):
        sexa = pyasl.coordsDegToSexa(UVBrightRA1[i],UVBrightDec1[i])
        sexaLen = len(sexa)-1
        coordArrayRA1.append(sexa[0:12])
        coordArrayDec1.append(sexa[14:sexaLen])

    coordArrayRA = np.array(coordArrayRA)
    coordArrayDec = np.array(coordArrayDec)
    coordArrayRA1 = np.array(coordArrayRA1)
    coordArrayDec1 = np.array(coordArrayDec1)

    #fileArray = np.column_stack((indAll[UVBrightCond],dat2["col2"][indAll][UVBrightCond],dat2["col3"][indAll][UVBrightCond],bv[indAll][UVBrightCond],v[indAll][UVBrightCond]))
    #fileArray1 = np.column_stack((indAll[UVBrightCond],coordArrayRA,coordArrayDec,bv[indAll][UVBrightCond],v[indAll][UVBrightCond]))
    #breakpoint()
    #infiles = np.savetxt("candStars/%s_candStars.dat"%(clusterName,ext),fileArray, fmt="%d,%.6f,%.6f,%.2f,%.2f", delimiter = ",",header="SiegelIndex,RA,Dec,(B-V)0,Mv")
    #infiles1 = np.savetxt("candStars/%s_candStars1.dat"%(clusterName,ext),fileArray1, fmt="%s", delimiter = "\t",header="Siegel_Index    RA    Dec    (B-V)_0    M_V")

    fig, ax = plt.subplots()
    ax.scatter(bv[indAll], v[indAll], c='k', s=0.1)
    # ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv[indAll].min(), bv[indAll].max(), len(bv[indAll]))
    # x_array = np.linspace(-0.6,1.3,100)
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35,0.05)
    ax.plot(xplot, y, "r-")
    ax.vlines(x=-0.05, ymin=-4, ymax=5)
    # ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.scatter(bv[indAll][UVBrightCond],v[indAll][UVBrightCond],c='g',s=2)
    ax.set_xlim(-0.75,1.6)
    ax.set_ylim(5,-4)
    #ax.text(1,4.2, "{}\n E(B-V) = {}".format(clusterName,ebv),bbox={'facecolor': 'white','alpha':0.5}, fontsize=10)
    ax.set_xlabel('($B-V$)$_0$')
    #ax.set_xlabel('(\textit{$B-V$})$_0$')
    ax.set_ylabel('M$_V$', style="italic")
    #ax.legend(loc="lower right", fontsize = "small")
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName,ebv,distModulus))
    #fig.savefig('Plots/DBSCAN/VBV/VvsBV_%s_DBSCAN.png'%(clusterName,ext),bbox_inches='tight',dpi=300)
    fig.savefig('Plots/DBSCAN/VBV/VvsBV_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)    
    #breakpoint()

    ##########################################IMPORTANT Non-Member CMDS#####################################################
    bvRaw = bRaw - vRaw
    fig, ax = plt.subplots()
    ax.scatter(bvRaw[notInClustSiegel], vRaw[notInClustSiegel], c='k', s=0.1)
    ax.set_xlim(-0.75, 1.6)
    ax.set_ylim(22, 10)
    ax.set_xlabel('($B-V$)$_0$')
    ax.set_ylabel('M$_V$', style="italic")
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName, ebv, distModulus))
    # xplot = np.linspace(bvRaw[indAll].min(), bvRaw[indAll].max(), len(bv[indAll]))
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    # ax.plot(xplot, y, "r-")
    ax.vlines(x=0.4, ymin=8, ymax=25, color="orangered", linestyle="--")
    fig.savefig('Plots/DBSCAN/nonMemberPlots/VBV/VvsBV_%s_NM.png'%(clusterName),bbox_inches='tight',dpi=300)

    fig, ax = plt.subplots()
    ax.scatter(bvRaw[notInClustSiegel], uRaw[notInClustSiegel], c='k', s=0.1)
    ax.set_xlim(-0.75, 1.6)
    ax.set_ylim(22, 10)
    ax.set_xlabel('($B-V$)$_0$')
    ax.set_ylabel('M$_u$', style="italic")
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName, ebv, distModulus))
    # xplot = np.linspace(bvRaw[indAll].min(), bvRaw[indAll].max(), len(bv[indAll]))
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.35, -0.15)
    # ax.plot(xplot, y, "r-")
    ax.vlines(x=0.4, ymin=8, ymax=25, color="orangered", linestyle="--")
    fig.savefig('Plots/DBSCAN/nonMemberPlots/UBV/UvsBV_%s_NM.png'%(clusterName), bbox_inches='tight', dpi=300)
    ##########################################IMPORTANT Non-Member CMDS#####################################################


    '''
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
    fig.savefig('Plots/DBSCAN/BBV/BvsBV_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)
    '''
    ##########################################IMPORTANT u-BAND CMDs#####################################################

    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],u[indAll],c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    x_array = np.linspace(-0.6,1.3,100)
    #y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1)
    xplot = np.linspace(bv[indAll].min(), bv[indAll].max(), len(bv[indAll]))
    # x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 1.65,0.8)
    # y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.83)
    ax.plot(xplot, y, "r-")
    ax.vlines(x=-0.05, ymin=-4, ymax=5)
    # ax.plot(xplot,model_f(xplot,*popt),'r--')
    # y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 1.65,-0.01)
    #ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    #y2 = model_f(bv[indAll], -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1)
    #UVBrightCond = np.logical_and(u[indAll]<y2,bv[indAll]<-0.05)
    UVBrightCond1 = np.logical_and(u[indAll] < y_u, bv[indAll] < -0.05)
    ax.scatter(bv[indAll][UVBrightCond1],u[indAll][UVBrightCond1],c='g',s=2)
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName, ebv, distModulus))
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(5,-4)
    ax.set_xlabel('($B-V$)$_0$')
    ax.set_ylabel('$M_u$')
    fig.savefig('Plots/DBSCAN/UBV/UvsBV_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

    UVBrightRA2 = dat2["col2"][indAll][UVBrightCond1]
    UVBrightDec2 = dat2["col3"][indAll][UVBrightCond1]

    coordArrayRA2 = []
    coordArrayDec2 = []

    for i in range(0, len(UVBrightRA2)):
        sexa = pyasl.coordsDegToSexa(UVBrightRA2[i], UVBrightDec2[i])
        sexaLen = len(sexa) - 1
        coordArrayRA2.append(sexa[0:12])
        coordArrayDec2.append(sexa[14:sexaLen])

    coordArrayRA2 = np.array(coordArrayRA2)
    coordArrayDec2 = np.array(coordArrayDec2)

    UVBrightRA3 = dat2["col2"][notInClustSiegel]
    UVBrightDec3 = dat2["col3"][notInClustSiegel]

    coordArrayRA3 = []
    coordArrayDec3 = []

    for i in range(0, len(UVBrightRA3)):
        sexa = pyasl.coordsDegToSexa(UVBrightRA3[i], UVBrightDec3[i])
        sexaLen = len(sexa) - 1
        coordArrayRA3.append(sexa[0:12])
        coordArrayDec3.append(sexa[14:sexaLen])

    coordArrayRA3 = np.array(coordArrayRA3)
    coordArrayDec3 = np.array(coordArrayDec3)
    ##########################################IMPORTANT u-BAND CMDs#####################################################

    indexColumn = indAll[UVBrightCond]+1
    indexColumn2 = indAll[UVBrightCond1]+1
    indexColumn3 = notInClustSiegel+1
    indexColumn1 = indAll+1
    flagArray1 = flagArray[indAll]
    flagArray2 = flagArray[indAll][UVBrightCond1]
    flagArray = flagArray[indAll][UVBrightCond]
    #flagArray3 = flagArray[notInClustSiegel]
    #fileArray = np.column_stack((indexColumn,dat2["col2"][indAll][UVBrightCond],dat2["col3"][indAll][UVBrightCond],bv[indAll][UVBrightCond],u[indAll][UVBrightCond],b[indAll][UVBrightCond],v[indAll][UVBrightCond],uRaw[indAll][UVBrightCond],bRaw[indAll][UVBrightCond],vRaw[indAll][UVBrightCond],iRaw[indAll][UVBrightCond],flagArray))
    #fileArray1 = np.column_stack((indexColumn,coordArrayRA,coordArrayDec,bv[indAll][UVBrightCond],u[indAll][UVBrightCond],b[indAll][UVBrightCond],v[indAll][UVBrightCond],uRaw[indAll][UVBrightCond],bRaw[indAll][UVBrightCond],vRaw[indAll][UVBrightCond],iRaw[indAll][UVBrightCond]))
    fileArray2 = np.rec.fromarrays([indexColumn,coordArrayRA,coordArrayDec,bv[indAll][UVBrightCond],u[indAll][UVBrightCond],b[indAll][UVBrightCond],v[indAll][UVBrightCond],uRaw[indAll][UVBrightCond],bRaw[indAll][UVBrightCond],vRaw[indAll][UVBrightCond],iRaw[indAll][UVBrightCond],flagArray])
    fileArray2 = np.array(fileArray2)
    fileArray3 = np.rec.fromarrays([indexColumn1,coordArrayRA1,coordArrayDec1,bv[indAll],u[indAll],b[indAll],v[indAll],uRaw[indAll],bRaw[indAll],vRaw[indAll],iRaw[indAll],flagArray1])
    fileArray4 = np.rec.fromarrays([indexColumn2,coordArrayRA2,coordArrayDec2,bv[indAll][UVBrightCond1],u[indAll][UVBrightCond1],b[indAll][UVBrightCond1],v[indAll][UVBrightCond1],uRaw[indAll][UVBrightCond1],bRaw[indAll][UVBrightCond1],vRaw[indAll][UVBrightCond1],iRaw[indAll][UVBrightCond1],flagArray2])
    fileArray4 = np.array(fileArray4)
    fileArray5 = np.rec.fromarrays([indexColumn3,coordArrayRA3,coordArrayDec3,bv[notInClustSiegel],u[notInClustSiegel],b[notInClustSiegel],v[notInClustSiegel],uRaw[notInClustSiegel],bRaw[notInClustSiegel],vRaw[notInClustSiegel], iRaw[notInClustSiegel]])
    fileArray5 = np.array(fileArray5)

    if len(fileArray2) != 0 and len(fileArray4) != 0:
        candStarMasterList = np.concatenate((fileArray2, fileArray4))
        candStarMasterList = np.unique(candStarMasterList)
    elif len(fileArray2) != 0 and len(fileArray4) == 0:
        candStarMasterList = fileArray2
    elif len(fileArray2) == 0 and len(fileArray4) != 0:
        candStarMasterList = fileArray4
    else:
        candStarMasterList = np.array([])

    #infiles = np.savetxt("candStars/CSVs/%s_candStars_%s.dat"%(clusterName,ext),fileArray, fmt="%d,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%i", delimiter = ",",header="#,RA,Dec,(B-V)0,Mu,Mb,Mv,u,B,V,I,BlueFlag,E(B-V)={:.2f}, m-M = {:.2f},BlueFlag Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then BlueFlag = 0".format(ebv,distModulus))
    #infiles1 = np.savetxt("candStars/TSVs/%s_candStars_%s.dat"%(clusterName,ext),fileArray2, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter = "\t",header="{} E(B-V)={:.2f}, m-M = {:.2f}\nBlueFlag Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then BlueFlag = 0\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag".format(clusterName,ebv, distModulus))
    infiles2 = np.savetxt("clusterMembers/%s_memberStars_%s.dat"%(clusterName,ext),fileArray3, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter = "\t",header="{} E(B-V)={:.2f}, (m-M)_0 = {:.2f}\nBlueFlag Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then BlueFlag = 0\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag".format(clusterName,ebv, distModulus))
    #infiles3 = np.savetxt("candStars/TSVs/uBVCandStars/%s_uBVcandStars_%s.dat"%(clusterName,ext),fileArray4, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter = "\t",header="{} E(B-V)={:.2f}, m-M = {:.2f}\nBlueFlag Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then BlueFlag = 0\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag".format(clusterName,ebv, distModulus))
    infiles4 = np.savetxt("nonMembers/%s_nonMembers_%s.dat"%(clusterName, ext),fileArray5, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f", delimiter = "\t",header="{} E(B-V)={:.2f}, (m-M)_0 = {:.2f}\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI".format(clusterName,ebv, distModulus))
    infiles5 = np.savetxt("candStars/candStarMasterList/%s_candStarsMaster_%s.dat"%(clusterName,ext),candStarMasterList, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter = "\t",header="{} E(B-V)={:.2f}, (m-M)_0 = {:.2f}\nBlueFlag Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then BlueFlag = 0\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag".format(clusterName,ebv, distModulus))

    '''
    fig, ax = plt.subplots()
    ax.scatter(bv[indAll],ub[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/DBSCAN/UBBV/UBvsBV_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)
    
    fig, ax = plt.subplots()
    ax.scatter(vi[indAll],ub[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/DBSCAN/UBVI/UBvsVI_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)
    
    fig, ax = plt.subplots()
    ax.scatter(vi[indAll],ubbv[indAll],c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('(U-B)-(B-V)')
    fig.savefig('Plots/DBSCAN/UBBVVI/UBBVvsVI_%s_DBSCAN_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)
    '''
def VBVplotBestFit(bv,v,clusterName,dist,ebv):
    distModulus = 5*np.log10(dist*100)
    fig, ax = plt.subplots()
    ax.scatter(bv,v,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2,0)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -4, ymax = 5)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.6)
    ax.set_ylim(5,-4)
    ax.set_xlabel('($B-V$)$_0$')
    ax.set_ylabel('$M_V$')
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName,ebv,distModulus))
    
    fig.savefig('Plots/VBV/VvsBV_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

def BBVplotBestFit(bv,b,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,b,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot-0.2, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2,0)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('B')
    fig.savefig('Plots/BBV/BvsBV_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)


def UBVplotBestFit(bv,u,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,u,c='k',s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 2.1,0)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U')
    fig.savefig('Plots/UBV/UvsBV_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

#Plots that don't need fitting

def UBBVplotBestFit(bv,ub,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(bv,ub,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('B-V')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/UBBV/UBvsBV_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

def UBVIplotBestFit(vi,ub,clusterName):
    fig, ax = plt.subplots()
    ax.scatter(vi,ub,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('U-B')
    fig.savefig('Plots/UBVI/UBvsVI_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

def UBBVVIplotBestFit(vi, ubbv, clusterName):
    fig, ax = plt.subplots()
    ax.scatter(vi,ubbv,c='k',s=0.1)
    ax.vlines(x = -0.05, ymin = -3, ymax = 4)
    ax.set_xlim(-0.75,1.3)
    ax.set_ylim(4,-3)
    ax.set_xlabel('V-I')
    ax.set_ylabel('(U-B)-(B-V)')
    fig.savefig('Plots/UBBVVI/UBBVvsVI_%s_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

def dereddenPlot(bvRAW,vRAW,bv,v,clusterName,dist,ebv):
    distModulus = 5*np.log10(dist*100)
    fig, ax = plt.subplots()
    ax.scatter(bvRAW,vRAW,c='k',s=0.1)
    ax.scatter(bv,v,c="r",s=0.1)
    #ax.scatter(bvhb,vhb,c='b',s=2)
    xplot = np.linspace(bv.min(),bv.max(),100001)
    #x_array = np.linspace(-0.6,1.3,100)
    y = model_f(xplot, -3.74, 7.03, 6.83, -19.86, 8.98, -1.51, 0.2,0)
    ax.plot(xplot, y, "r-")
    ax.vlines(x = -0.05, ymin = -4, ymax = 5)
    #ax.plot(xplot,model_f(xplot,*popt),'r--')
    #ax.set_xlim(-0.75,1.6)
    ax.set_ylim(5,-4)
    ax.set_xlabel('($B-V$)$_0$')
    ax.set_ylabel('$M_V$')
    ax.set_title("{} $E(B-V)$={:.2f} $(m-M)_0$={:.2f}".format(clusterName,ebv,distModulus))
    
    fig.savefig('Plots/deRed/VvsBV_%s.png'%(clusterName,ext),bbox_inches='tight',dpi=300)

def model_f(x,a,b,c,d,e,f,g,k):
    x=x+k
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
    cond = np.logical_and.reduce((b<60,v<60, chi<3, abs(sharp)<0.5))
    #cond = np.logical_and.reduce((b<60,v<60))
    ind = np.where(cond)[0]

    # De-redden the cluster
    u2, b2, v2, i2 = dereddening(u, b, v, i, ebv, ru, rb, rv, ri, dist)
    bv = b2-v2 # Define a color
    vi = v2-i2
    ub = u2-b2
    ubbv = (u2-b2)-(b2-v2)

    blueFlagArray = []
    for j in range(0,len(bv)):
        if (vi[j] > 0.331+1.444*bv[j]):
            blueFlagArray.append(0)
        else:
            blueFlagArray.append(1)
    blueFlagArray = np.array(blueFlagArray)

    # VBVplotBestFit(bv,v2,clusterName,dist,ebv)
    # UBVplotBestFit(bv,u2,clusterName)
    # UBBVplotBestFit(bv,ub,clusterName)
    # UBVIplotBestFit(vi,ub,clusterName)
    # UBBVVIplotBestFit(vi,ubbv,clusterName)
    # BBVplotBestFit(bv,b2,clusterName)
    DBSCANPlots(bv,v2,b2,u2,vi,ub,ubbv,clusterName, ind,dist,raClust,decClust,c,r_c,u,v,b,i,ebv,blueFlagArray)
    # You're done for this cluster!

def main():
    start = time.time()
    clusterFunc(dat, ebv, dist, rv=3.315,ru=5.231,rb=4.315, ri = 1.940)
    end = time.time()
    timeTaken = (end-start)
    logging.info('Run successful, extension: {}, execution time: {:.2f}'.format(ext, timeTaken))
    print("Execution time: ", timeTaken)


if __name__ == "__main__":
    logging.basicConfig(filename='logs.log',
                        encoding='utf-8',
                        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO)

    clusterName = input("Enter a cluster name:")
    clusterNameFile = ("{}.phot".format(clusterName))
    clusterGaiaFile = ("NewCoor/{}_new_coor.dat".format(clusterName))
    clusterGaiaResults = ("{}.csv".format(clusterName))
    # clusterNameDict = clusterName.replace()
    # dist = float(input("Enter a distance:"))
    dist = float(locals()[clusterName]['dist'])
    # ebv = float(input("Enter a E(B-V) value:"))
    ebv = float(locals()[clusterName]['ebv'])
    # raClust = str(input("Enter the RA with spaces:"))
    raClust = str(locals()[clusterName]['ra'])
    # decClust = str(input("Enter the Dec with spaces:"))
    decClust = str(locals()[clusterName]['dec'])
    # c = float(input("Enter King-model central concentration value:"))
    c = float(locals()[clusterName]['c'])
    # r_c = float(input("Enter a core radius in arcminutes:"))
    r_c = float(locals()[clusterName]['r_c'])

    dat = Table.read(clusterNameFile, format="ascii")
    dat2 = Table.read(clusterGaiaFile, format="ascii")

    dat4 = pd.read_csv("GAIAData/{}".format(clusterGaiaResults))
    dat4 = dat4.fillna(1000)

    unID, indexes, inverse, counts = np.unique(dat4[f"{clusterName.lower()}_oid"],
                                               return_index=True,
                                               return_counts=True,
                                               return_inverse=True)

    index2 = np.where(counts > 1)[0]
    oids = dat4[f"{clusterName.lower()}_oid"][index2]
    oids = np.asarray(oids)
    clusterOids = np.asarray(dat4[f"{clusterName.lower()}_oid"])
    angSep = np.asarray(dat4["ang_sep"])
    parallaxArray = np.asarray(dat4["parallax"])
    angSepUniq, parUniq = angSep[indexes], parallaxArray[indexes]

    indexKeep = np.zeros_like(counts)

    for i, oid in enumerate(unID):
        indexi = indexes[i]
        if counts[i] > 1:
            angList, indList = [], []
            for j in range(counts[i]):
                if parUniq[i + j] > 90:
                    continue
                else:
                    try:
                        angList.append(angSepUniq[i + j])
                        indList.append(indexi + j)
                    except:
                        pass
            # breakpoint()
            if len(angList) == 0:
                indexKeep[i] = indexi
            else:
                angList, indList = np.array(angList), np.array(indList)
                indexKeep[i] = indList[np.argmin(angList)]
        else:
            indexKeep[i] = indexi

    PMRA = pd.DataFrame(dat4["pmra"][indexKeep])
    PMDEC = pd.DataFrame(dat4["pmdec"][indexKeep])
    oids = pd.DataFrame(dat4[f"{clusterName.lower()}_oid"][indexKeep])
    parallaxGaia = pd.DataFrame(dat4["parallax"][indexKeep])
    parallaxErrorGaia = pd.DataFrame(dat4["parallax_error"][indexKeep])
    dat5 = pd.concat([PMRA, PMDEC, oids, parallaxGaia, parallaxErrorGaia], axis="columns")

    main()
