import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
from PyAstronomy import pyasl
from clusterProps import clusterProperties as CP
from GaiaQuerier import GaiaQuerier
from GMagCalculator import GMagFinder
from GCDBSCAN import GCDBSCAN
from datetime import datetime
import logging
import os
import json
from GCAnalyzerHelperFunctions import *
from baumgardtComparer import Comparer


# Constants


# Extension (to keep track of generated files)
# extension = datetime.now().strftime('%B') - gets month as a string, e.g. "April"

class GCAnalyzer:
    def __init__(self):
        self.clusterName = input("Enter a cluster name:")
        if self.clusterName == "":
            exit("No cluster specified")

        ext = input("Use current month as extension for file tracking? [Y/N]")
        if ext in ["Y", "y", ""]:
            self.extension = datetime.now().strftime('%B')
        else:
            self.extension = input("Enter custom extension: ")
        print(f"Cluster Analysis Started on {self.clusterName} with extension: {self.extension}")

        self.clusterNameFile = f"{self.clusterName}.phot"
        self.clusterGaiaFile = f"NewCoor/{self.clusterName}_new_coor.dat"
        self.clusterGMagFile = f"gData/{self.clusterName}.csv"
        self.clusterGaiaResults = f"{self.clusterName}.csv"
        self.dist = float(CP[self.clusterName]['dist'])  # Distance to the GC [kpc] (Harris)
        self.ebv = float(CP[self.clusterName]['ebv'])  # GC Reddening (Harris)
        self.raClust = str(CP[self.clusterName]['ra'])  # Cluster RA (Harris)
        self.decClust = str(CP[self.clusterName]['dec'])  # Cluster Dec (Harris)
        self.c = float(CP[self.clusterName]['c'])  # Cluster King-model central concentration (Harris)
        self.r_c = float(CP[self.clusterName]['r_c'])  # Cluster core radius (Harris)
        self.distModulus = float(CP[self.clusterName]['harrisDistMod'])  # Cluster distance modulus (Harris)

        try:
            self.photometry_data = Table.read(self.clusterNameFile, format="ascii")  # Siegel File
            self.gaia_coords_data = Table.read(self.clusterGaiaFile, format="ascii")  # Gaia New_Coor file
            print("-->Photometry/Gaia Coordinates file found")

            self.u = self.photometry_data['col10']
            self.b = self.photometry_data['col4']
            self.v = self.photometry_data['col8']
            self.i = self.photometry_data['col6']
            self.chi = self.photometry_data['col12']
            self.sharp = self.photometry_data['col13']
            self.ra = self.gaia_coords_data['col2']
            self.dec = self.gaia_coords_data['col3']
        except:
            print("ERROR: Photometry/Gaia Coordinates file not found!")
            exit("Analysis Ended")

        if os.path.exists(self.clusterGMagFile):
            print("-->G magnitudes file found")
        else:
            gMagQuery = input("-->G magnitudes file not found. Make G magnitudes file for this cluster? [Y/N]")
            if gMagQuery in ["Y", "y", ""]:
                GMagFinder(self.clusterName, self.b, self.v, self.ra, self.dec)
            else:
                exit("Analysis Ended")

        if os.path.exists(f"GAIAData/{self.clusterGaiaResults}"):
            print(f"-->Gaia DR3 matches file found")
        else:
            # print(f"-->Gaia DR3 matches file not found")
            makeQuery = input("-->Gaia DR3 matches file not found. Make Gaia Archive query for this cluster? [Y/N]")
            if makeQuery in ["Y", "y", ""]:
                print("-->Gaia DR3 Querying Started")
                GaiaQuerier(self.clusterName)
            else:
                exit("Analysis Ended")

        self.gaia_matches_data = pd.read_csv(f"GAIAData/{self.clusterGaiaResults}").fillna(1000)  # Gaia Matches
        # self.gaia_matches_data = self.gaia_matches_data.fillna(1000)

    def dataCleaner(self):
        # Remove missing magnitudes and bad fits (chi^2 and sharp features also)
        cond = np.logical_and.reduce((self.b < 60, self.v < 60, self.chi < 3, abs(self.sharp) < 0.5))
        self.initialCleanupCond = np.where(cond)[0]
        self.badPhotometry = np.where(cond == False)[0]

    def dereddening(self):
        #  This method performs dereddening on the cluster using Gautam's modeled extinction values (see text)
        rv = 3.315
        ru = 5.231
        rb = 4.315
        ri = 1.940
        self.bv = self.b - self.v
        self.dered_u = self.u - 5 * np.log10(self.dist * 100) - (
                ru * self.ebv - 0.00341 * self.bv - 0.0131 * self.ebv * self.bv)
        self.dered_b = self.b - 5 * np.log10(self.dist * 100) - (
                rb * self.ebv - 0.0454 * self.bv - 0.142 * self.ebv * self.bv)
        self.dered_v = self.v - 5 * np.log10(self.dist * 100) - (
                rv * self.ebv - 0.0143 * self.bv - 0.0568 * self.ebv * self.bv)
        self.dered_i = self.i - 5 * np.log10(self.dist * 100) - ri * self.ebv
        self.dered_bv = self.dered_b - self.dered_v
        self.dered_vi = self.dered_v - self.dered_i
        self.dered_ub = self.dered_u - self.dered_b
        self.dered_ubbv = self.dered_ub - self.dered_bv

        #  This is a simple check to see if a given star is blue or not by Howard's definition
        self.blueFlagArray = []
        for j in range(0, len(self.bv)):
            if self.dered_vi[j] > 0.331 + 1.444 * self.dered_bv[j]:
                self.blueFlagArray.append(0)
            else:
                self.blueFlagArray.append(1)
        self.blueFlagArray = np.array(self.blueFlagArray)

    # def tidalRadius(self):
    #     self.tidal_radius = self.r_c * (10 ** self.c)

    def uniqueMatches(self):
        unID, indexes, inverse, counts = np.unique(self.gaia_matches_data[f"{self.clusterName.lower()}_oid"],
                                                   return_index=True,
                                                   return_counts=True,
                                                   return_inverse=True)

        index2 = np.where(counts > 1)[0]
        oids = self.gaia_matches_data[f"{self.clusterName.lower()}_oid"][index2]
        oids = np.asarray(oids)
        clusterOids = np.asarray(self.gaia_matches_data[f"{self.clusterName.lower()}_oid"])
        angSep = np.asarray(self.gaia_matches_data["ang_sep"])
        parallaxArray = np.asarray(self.gaia_matches_data["parallax"])
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

        PMRA = pd.DataFrame(self.gaia_matches_data["pmra"][indexKeep])
        PMDEC = pd.DataFrame(self.gaia_matches_data["pmdec"][indexKeep])
        oids = pd.DataFrame(self.gaia_matches_data[f"{self.clusterName.lower()}_oid"][indexKeep])
        parallaxGaia = pd.DataFrame(self.gaia_matches_data["parallax"][indexKeep])
        parallaxErrorGaia = pd.DataFrame(self.gaia_matches_data["parallax_error"][indexKeep])

        self.unique_gaia_matches = pd.concat([PMRA, PMDEC, oids, parallaxGaia,
                                              parallaxErrorGaia], axis="columns")  # Gaia matches with extra information

        # Begin DBSCAN clustering here
        DBSCANClustering = GCDBSCAN(self.unique_gaia_matches, self.clusterName, self.extension)
        DBSCANClustering.PMPlots()
        DBSCANClustering.clusters()
        DBSCANClustering.clusterPlots()
        self.indSiegel, self.indAll1, self.notInClustInd = DBSCANClustering.indexOrganizer(self.initialCleanupCond)
        # Called self.indAll self._ here to make sure it does not conflict with further definitions

    def clusterMembershipChecks(self):
        clusterCore = coordTransfer(self.raClust, self.decClust)  # Find the decimal coordinates for the cluster core
        angSep = arcMinAngularSep(clusterCore[0], clusterCore[1], self.gaia_coords_data["col2"],
                                  self.gaia_coords_data["col3"])  # Calculate the angular sep between each star and core
        parStarErr, parDiff = parallaxError(self.dist, self.unique_gaia_matches["parallax"],
                                            self.unique_gaia_matches["parallax_error"])  # Get the plx error and diff
        parCond = np.where(parDiff < 5 * parStarErr)[0]  # Check to see which stars have plx+/5 sigma of the cluster plx
        self.tidal_radius = self.r_c * (10 ** self.c)
        # tidalRad = tidalRadius(self.c, self.r_c)  # Get the tidal radius of the cluster using Harris c and r_c
        tidalCond = np.where(angSep < self.tidal_radius)[0]  # Check to see if the star is within the tidal radius
        self.indAll = np.intersect1d(self.indAll1, tidalCond)
        indParSiegel = self.indSiegel[parCond]
        self.indAll = np.intersect1d(self.indAll, indParSiegel)  # Get combined index of members


        # Getting non-members from membership tests (bad photometry rejects are in dataCleaner method)

        counter = []  # Initialize list for reason for exclusion from cluster membership

        self.notInTidalRadCond = np.where(angSep > self.tidal_radius)[0]
        self.notInParCond = np.where(parDiff > 5 * parStarErr)[0]
        self.notInParCond = self.indSiegel[self.notInParCond]
        # notInClust = np.where(labels != 0)[0]
        self.notInClust = self.indSiegel[self.notInClustInd]

        nonMemberCond = np.concatenate((self.notInTidalRadCond, self.notInParCond, self.notInClust, self.badPhotometry))
        nonMemberCond = np.unique(nonMemberCond)

        self.notInClustSiegel = np.intersect1d(self.indSiegel, nonMemberCond)

        for star in self.notInClustSiegel:
            if star in self.badPhotometry:
                counter.append(0)
            elif star in self.notInClust:
                counter.append(1)
            elif star in self.notInTidalRadCond:
                counter.append(2)
            elif star in self.notInParCond:
                counter.append(3)

        self.counter = np.array(counter)  # Get combined list of reasons for exclusion for each star

    def HBParams(self):
        # This function sets the HB model parameters. To change the parameters, change the .json file directly
        HBParamsPath = f"HBParams/UBV/HBParams_UBV_{self.clusterName}_{self.extension}.json"
        if os.path.exists(HBParamsPath):
            print("-->Optimal horizontal branch model parameters found")
            with open(f"HBParams/UBV/HBParams_UBV_{self.clusterName}_{self.extension}.json", 'r') as f:
                self.UBV_HBParams = json.load(f)
            with open(f"HBParams/VBV/HBParams_VBV_{self.clusterName}_{self.extension}.json", 'r') as f:
                self.VBV_HBParams = json.load(f)
        else:
            print("-->Optimal horizontal branch model parameters not found")
            defaultHBQuery = input("-->Use default (M 79) parameters? [Y/N]")  # Defaults to M79 parameters
            if defaultHBQuery in ["Y", "y", ""]:
                with open(f"HBParams/UBV/HBParams_UBV_M79.json", 'r') as f:
                    self.UBV_HBParams = json.load(f)
                with open(f"HBParams/VBV/HBParams_VBV_M79.json", 'r') as f:
                    self.VBV_HBParams = json.load(f)
            else:
                exit("Analysis Ended")

    def CMDPlotter(self):
        try:
            print("-->CMD plotting started")
            # V vs B-V HB model

            HB_VBV = HB_model(self.dered_bv[self.indAll], float(self.VBV_HBParams['a']), float(self.VBV_HBParams['b']),
                              float(self.VBV_HBParams['c']), float(self.VBV_HBParams['d']),
                              float(self.VBV_HBParams['e']),
                              float(self.VBV_HBParams['f']), float(self.VBV_HBParams['g']),
                              float(self.VBV_HBParams['k']))

            # u vs B-V HB model
            HB_UBV = HB_model(self.dered_bv[self.indAll], float(self.UBV_HBParams['a']), float(self.UBV_HBParams['b']),
                              float(self.UBV_HBParams['c']), float(self.UBV_HBParams['d']),
                              float(self.UBV_HBParams['e']),
                              float(self.UBV_HBParams['f']), float(self.UBV_HBParams['g']),
                              float(self.UBV_HBParams['k']))

            # UV Bright conditions for V vs B-V and u vs B-V CMDs. These use the *dereddened* values
            self.UVBrightCond_VBV = np.logical_and(self.dered_v[self.indAll] < HB_VBV,
                                                   self.dered_bv[self.indAll] < -0.05)
            self.UVBrightCond_UBV = np.logical_and(self.dered_u[self.indAll] < HB_UBV,
                                                   self.dered_bv[self.indAll] < -0.05)

            # The following CMDs are made using the observed magnitudes and colors, not the dereddened ones
            # V vs B-V Plot
            fig, ax = plt.subplots()
            ax.scatter(self.bv[self.indAll], self.v[self.indAll], c='k', s=0.1)
            ax.vlines(x=-0.05, ymin=-4, ymax=5)
            ax.scatter(self.bv[self.indAll][self.UVBrightCond_VBV], self.v[self.indAll][self.UVBrightCond_VBV], c='g',
                       s=2)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            ax.set_xlim(-0.75, 1.3)
            ax.set_ylim(21, 11.5)
            ax.set_xlabel('$B-V$', fontsize=14)
            ax.set_ylabel('$V$', fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            fig.savefig(f'Plots/DBSCAN/RAW/VBV/VvsBV_{self.clusterName}_DBSCAN_RAW_{self.extension}.pdf',
                        bbox_inches='tight', dpi=300)

            # u vs B-V Plot
            fig, ax = plt.subplots()
            ax.scatter(self.bv[self.indAll], self.u[self.indAll], c='k', s=0.1)
            ax.vlines(x=-0.05, ymin=-4, ymax=5)
            ax.scatter(self.bv[self.indAll][self.UVBrightCond_UBV], self.u[self.indAll][self.UVBrightCond_UBV], c='g',
                       s=2)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            ax.set_xlim(-0.75, 1.3)
            ax.set_ylim(21, 13.5)
            ax.set_xlabel('$B-V$', fontsize=14)
            ax.set_ylabel('$u$', fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            fig.savefig(f'Plots/DBSCAN/RAW/UBV/UvsBV_{self.clusterName}_DBSCAN_RAW_{self.extension}.pdf',
                        bbox_inches='tight', dpi=300)
            print("---->Observed CMDs plotted")

            # The following CMDs are made using the dereddened magnitudes and colors
            xplot = np.linspace(self.dered_bv[self.indAll].min(), self.dered_bv[self.indAll].max(),
                                len(self.dered_bv[self.indAll]))

            # V vs B-V Plot
            fig, ax = plt.subplots()
            ax.scatter(self.dered_bv[self.indAll], self.dered_v[self.indAll], c='k', s=0.1)
            ax.plot(xplot, HB_model(xplot, float(self.VBV_HBParams['a']), float(self.VBV_HBParams['b']),
                                    float(self.VBV_HBParams['c']), float(self.VBV_HBParams['d']),
                                    float(self.VBV_HBParams['e']),
                                    float(self.VBV_HBParams['f']), float(self.VBV_HBParams['g']),
                                    float(self.VBV_HBParams['k'])), "r-")
            ax.vlines(x=-0.05, ymin=-4, ymax=5)
            ax.scatter(self.dered_bv[self.indAll][self.UVBrightCond_VBV],
                       self.dered_v[self.indAll][self.UVBrightCond_VBV],
                       c='g', s=2)
            ax.set_xlim(-0.75, 1.6)
            ax.set_ylim(5, -4)
            ax.set_xlabel('($B-V$)$_0$', fontsize=14)
            ax.set_ylabel('M$_V$', style="italic", fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            fig.savefig(f'Plots/DBSCAN/VBV/VvsBV_{self.clusterName}_DBSCAN_{self.extension}.pdf', bbox_inches='tight',
                        dpi=300)

            # u vs B-V Plot
            fig, ax = plt.subplots()
            ax.scatter(self.dered_bv[self.indAll], self.dered_u[self.indAll], c='k', s=0.1)
            ax.plot(xplot, HB_model(xplot, float(self.UBV_HBParams['a']), float(self.UBV_HBParams['b']),
                                    float(self.UBV_HBParams['c']), float(self.UBV_HBParams['d']),
                                    float(self.UBV_HBParams['e']),
                                    float(self.UBV_HBParams['f']), float(self.UBV_HBParams['g']),
                                    float(self.UBV_HBParams['k'])), "r-")
            ax.vlines(x=-0.05, ymin=-4, ymax=5)
            ax.vlines(x=-0.05, ymin=-3, ymax=4)
            ax.scatter(self.dered_bv[self.indAll][self.UVBrightCond_UBV],
                       self.dered_u[self.indAll][self.UVBrightCond_UBV],
                       c='g', s=2)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            ax.set_xlim(-0.75, 1.3)
            ax.set_ylim(5, -4)
            ax.set_xlabel('($B-V$)$_0$', fontsize=14)
            ax.set_ylabel('$M_u$', fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            fig.savefig(f'Plots/DBSCAN/UBV/UvsBV_{self.clusterName}_DBSCAN_{self.extension}.pdf', bbox_inches='tight',
                        dpi=300)
            print("---->Dereddened CMDs plotted")

            # The following are the CMDs for the non-member stars
            fig, ax = plt.subplots()
            ax.scatter(self.bv[self.notInClustSiegel], self.v[self.notInClustSiegel], c='k', s=0.1)
            ax.set_xlim(-0.75, 1.6)
            ax.set_ylim(22, 10)
            ax.set_xlabel('($B-V$)$_0$', fontsize=14)
            ax.set_ylabel('M$_V$', style="italic", fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            ax.vlines(x=0.4, ymin=8, ymax=25, color="orangered", linestyle="--")
            fig.savefig(f'Plots/DBSCAN/nonMemberPlots/VBV/VvsBV_{self.clusterName}_NM.pdf', bbox_inches='tight',
                        dpi=300)

            # UBV (Non-Member) Plots

            fig, ax = plt.subplots()
            ax.scatter(self.bv[self.notInClustSiegel], self.u[self.notInClustSiegel], c='k', s=0.1)
            ax.set_xlim(-0.75, 1.6)
            ax.set_ylim(22, 10)
            ax.set_xlabel('($B-V$)$_0$', fontsize=14)
            ax.set_ylabel('M$_u$', style="italic", fontsize=14)
            plt.tick_params(axis='y', which='major', labelsize=14)
            plt.tick_params(axis='x', which='major', labelsize=14)
            ax.set_title(f"{self.clusterName} $E(B-V)$={self.ebv:.2f} $(m-M)_0$={self.distModulus:.2f}", fontsize=16)
            ax.vlines(x=0.4, ymin=8, ymax=25, color="orangered", linestyle="--")
            fig.savefig(f'Plots/DBSCAN/nonMemberPlots/UBV/UvsBV_{self.clusterName}_NM.pdf', bbox_inches='tight',
                        dpi=300)
            print("---->Non-member CMDs plotted")

            print("-->CMD plotting completed")
        except:
            exit("Error in plotting CMDs")

        self.UVBrightRA_VBV = self.gaia_coords_data["col2"][self.indAll][self.UVBrightCond_VBV]
        self.UVBrightDec_VBV = self.gaia_coords_data["col3"][self.indAll][self.UVBrightCond_VBV]
        # UVBright_VBV_Coords = list(zip(UVBrightRA_VBV, UVBrightDec_VBV))

        self.UVBrightRA_UBV = self.gaia_coords_data["col2"][self.indAll][self.UVBrightCond_UBV]
        self.UVBrightDec_UBV = self.gaia_coords_data["col3"][self.indAll][self.UVBrightCond_UBV]
        # UVBright_UBV_Coords = list(zip(UVBrightRA_UBV, UVBrightDec_UBV))

        memberRA = self.gaia_coords_data["col2"][self.indAll]
        memberDec = self.gaia_coords_data["col3"][self.indAll]

        nonMemberRA = self.gaia_coords_data["col2"][self.notInClustSiegel]
        nonMemberDec = self.gaia_coords_data["col3"][self.notInClustSiegel]

        coordArrayRA_VBV = []
        coordArrayDec_VBV = []
        coordArrayRA_UBV = []
        coordArrayDec_UBV = []
        coordArrayRA_Mem = []
        coordArrayDec_Mem = []
        coordArrayRA_NM = []
        coordArrayDec_NM = []

        for i in range(0, len(self.UVBrightRA_VBV)):
            sexa = pyasl.coordsDegToSexa(self.UVBrightRA_VBV[i], self.UVBrightDec_VBV[i])
            sexaLen = len(sexa) - 1
            coordArrayRA_VBV.append(sexa[0:12])
            coordArrayDec_VBV.append(sexa[14:sexaLen])

        for i in range(0, len(self.UVBrightRA_UBV)):
            sexa = pyasl.coordsDegToSexa(self.UVBrightRA_UBV[i], self.UVBrightDec_UBV[i])
            sexaLen = len(sexa) - 1
            coordArrayRA_UBV.append(sexa[0:12])
            coordArrayDec_UBV.append(sexa[14:sexaLen])

        for i in range(0, len(memberRA)):
            sexa = pyasl.coordsDegToSexa(memberRA[i], memberDec[i])
            sexaLen = len(sexa) - 1
            coordArrayRA_Mem.append(sexa[0:12])
            coordArrayDec_Mem.append(sexa[14:sexaLen])

        for i in range(0, len(nonMemberRA)):
            sexa = pyasl.coordsDegToSexa(nonMemberRA[i], nonMemberDec[i])
            sexaLen = len(sexa) - 1
            coordArrayRA_NM.append(sexa[0:12])
            coordArrayDec_NM.append(sexa[14:sexaLen])

        self.coordArrayRA_VBV = np.array(coordArrayRA_VBV)
        self.coordArrayDec_VBV = np.array(coordArrayDec_VBV)
        self.coordArrayRA_UBV = np.array(coordArrayRA_UBV)
        self.coordArrayDec_UBV = np.array(coordArrayDec_UBV)
        self.coordArrayRA_Mem = np.array(coordArrayRA_Mem)
        self.coordArrayDec_Mem = np.array(coordArrayDec_Mem)
        self.coordArrayRA_NM = np.array(coordArrayRA_NM)
        self.coordArrayDec_NM = np.array(coordArrayDec_NM)


    def dataSaver(self):
        indexColumn = self.indAll[self.UVBrightCond_VBV] + 1
        indexColumn2 = self.indAll[self.UVBrightCond_UBV] + 1
        indexColumn3 = self.notInClustSiegel + 1
        indexColumn1 = self.indAll + 1
        flagArray1 = self.blueFlagArray[self.indAll]
        flagArray2 = self.blueFlagArray[self.indAll][self.UVBrightCond_UBV]
        flagArray = self.blueFlagArray[self.indAll][self.UVBrightCond_VBV]

        # print(indexColumn)
        # breakpoint()
        indexColumn4 = np.unique(np.append(indexColumn, indexColumn2))
        uniqueInd = np.unique(np.append(self.indAll[self.UVBrightCond_VBV], self.indAll[self.UVBrightCond_UBV]))

        fileArray2 = np.rec.fromarrays(
            [indexColumn, self.coordArrayRA_VBV, self.coordArrayDec_VBV,
             self.dered_bv[self.indAll][self.UVBrightCond_VBV], self.dered_u[self.indAll][self.UVBrightCond_VBV],
             self.dered_b[self.indAll][self.UVBrightCond_VBV], self.dered_v[self.indAll][self.UVBrightCond_VBV],
             self.u[self.indAll][self.UVBrightCond_VBV], self.b[self.indAll][self.UVBrightCond_VBV],
             self.v[self.indAll][self.UVBrightCond_VBV], self.i[self.indAll][self.UVBrightCond_VBV], flagArray])
        fileArray2 = np.array(fileArray2)

        fileArray3 = np.rec.fromarrays(
            [indexColumn1, self.coordArrayRA_Mem, self.coordArrayDec_Mem, self.dered_bv[self.indAll],
             self.dered_u[self.indAll], self.dered_b[self.indAll], self.dered_v[self.indAll], self.u[self.indAll],
             self.b[self.indAll], self.v[self.indAll], self.i[self.indAll], flagArray1])

        fileArray4 = np.rec.fromarrays(
            [indexColumn2, self.coordArrayRA_UBV, self.coordArrayDec_UBV,
             self.dered_bv[self.indAll][self.UVBrightCond_UBV], self.dered_u[self.indAll][self.UVBrightCond_UBV],
             self.dered_b[self.indAll][self.UVBrightCond_UBV], self.dered_v[self.indAll][self.UVBrightCond_UBV],
             self.u[self.indAll][self.UVBrightCond_UBV], self.b[self.indAll][self.UVBrightCond_UBV],
             self.v[self.indAll][self.UVBrightCond_UBV], self.i[self.indAll][self.UVBrightCond_UBV], flagArray2])
        fileArray4 = np.array(fileArray4)

        fileArray5 = np.rec.fromarrays(
            [indexColumn3, self.coordArrayRA_NM, self.coordArrayDec_NM, self.dered_bv[self.notInClustSiegel],
             self.dered_u[self.notInClustSiegel], self.dered_b[self.notInClustSiegel],
             self.dered_v[self.notInClustSiegel], self.u[self.notInClustSiegel], self.b[self.notInClustSiegel],
             self.v[self.notInClustSiegel], self.i[self.notInClustSiegel], self.counter])
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

        print("-->Saving data files")

        np.savetxt(f"clusterMembers/{self.clusterName}_memberStars_{self.extension}.dat", fileArray3,
                   fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter="\t",
                   header=f"{self.clusterName} E(B-V)={self.ebv:.2f}, (m-M)_0 = {self.distModulus:.2f}\nBlueFlag "
                          f"Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then"
                          "BlueFlag = 0\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag")
        print("---->Member stars file saved")

        np.savetxt(f"nonMembers/{self.clusterName}_nonMembers_{self.extension}", fileArray5,
                   fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i", delimiter="\t",
                   header=f"{self.clusterName} E(B-V)={self.ebv:.2f}, (m-M)_0 = {self.distModulus:.2f}\nReason Note: "
                          f"0 -> Bad Photometry | 1 -> DBSCAN | 2 ->"
                          "Tidal Radius | 3-> Parallax\n\tRA\t\tDec\t\t("
                          "B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tReason")
        print("---->Non-member stars file saved")

        np.savetxt(f"candStars/candStarMasterList/{self.clusterName}_candStarsMaster_{self.extension}.dat",
                   candStarMasterList, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i",
                   delimiter="\t",
                   header=f"{self.clusterName} E(B-V)={self.ebv:.2f}, (m-M)_0 = {self.distModulus:.2f}\nBlueFlag "
                          f"Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then"
                          "BlueFlag = 0\n\tRA\t\tDec\t\t("
                          "B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag")
        print("---->Candidate stars file saved")

        if os.path.exists(f"candStars/candStarsWithProbs/{self.clusterName}_candStarsWithProb_{self.extension}.dat"):
            print("---->V&B21 Comaprison Data Found!")
        else:
            UVBrightRA = np.unique(np.append(np.array(self.UVBrightRA_VBV), np.array(self.UVBrightRA_UBV)))
            UVBrightDec = np.unique(np.append(np.array(self.UVBrightDec_VBV), np.array(self.UVBrightDec_UBV)))

            UVBright_Coords = list(zip(UVBrightRA, np.unique(UVBrightDec)))

            probFinder = Comparer(self.clusterName, UVBright_Coords)
            self.probList = probFinder.separationFinder()  # The probability of a star being a member from MNRAS, 505, 5978

            UVBrightCoordRA = []
            UVBrightCoordDec = []

            for i in range(0, len(UVBrightRA)):
                sexa = pyasl.coordsDegToSexa(UVBrightRA[i], UVBrightDec[i])
                sexaLen = len(sexa) - 1
                UVBrightCoordRA.append(sexa[0:12])
                UVBrightCoordDec.append(sexa[14:sexaLen])

            self.UVBrightCoordRA = np.array(UVBrightCoordRA)
            self.UVBrightCoordDec = np.array(UVBrightCoordDec)

            fileArray6 = np.rec.fromarrays(
                [indexColumn4, self.UVBrightCoordRA, self.UVBrightCoordDec,
                 self.dered_bv[uniqueInd], self.dered_u[uniqueInd],
                 self.dered_b[uniqueInd], self.dered_v[uniqueInd],
                 self.u[uniqueInd], self.b[uniqueInd],
                 self.v[uniqueInd], self.i[uniqueInd], self.blueFlagArray[uniqueInd], self.probList])
            fileArray6 = np.array(fileArray6)

            np.savetxt(f"candStars/candStarsWithProbs/{self.clusterName}_candStarsWithProb_{self.extension}.dat",
                       fileArray6, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\t%i\t%i",
                       delimiter="\t",
                       header=f"{self.clusterName} E(B-V)={self.ebv:.2f}, (m-M)_0 = {self.distModulus:.2f}\nBlueFlag "
                              f"Note: if (V-I)0 > 0.331 + 1.444 * (B-V)0 then"
                              "BlueFlag = 0\nNote: if BaumgardtCheck = 1, then there is atleast 1 member star within 3\" "
                              "of the candidate star in the cluster V&B21\n\tRA\t\tDec\t\t("
                              "B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI\tBlueFlag\tBaumgardtCheck")
            print("---->Candidate stars with probabilities file saved")

        print("-->All data files saved")
        print("Analysis Completed!")
        logging.info(f'Run successful, cluster: {self.clusterName}, extension: {self.extension}')
