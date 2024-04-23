import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import astropy.units as u
from tqdm import tqdm
import warnings

# Ignore VisibleDeprecationWarning
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Comparer:
    def __init__(self, clusterName, candStarCoordinates, bMagnitude, vMagnitude):
        self.clusterName = clusterName
        try:
            self.baumgardtData = pd.read_csv(f"BaumgardtTables/{self.clusterName}.txt", sep="\t",
                                             header=None)
        except Exception as e:
            exit(f"Baumgardt comparison file not found: {e}")

        self.b = bMagnitude
        self.v = vMagnitude
        self.g = self.v - 0.0124 * (self.b - self.v)
        baumgardtRA = self.baumgardtData[1]
        baumgardtDec = self.baumgardtData[2]
        self.baumgardtCoords = np.column_stack((baumgardtRA, baumgardtDec))
        self.baumgardtTree = cKDTree(self.baumgardtCoords)
        self.candidateList = candStarCoordinates

    def separationFinder(self):
        cadidateStarCoords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg) for ra, dec in self.candidateList]

        radius = 3 * u.arcsec
        matched_indices = []

        print("-->Comparing memberships to Vasiliev & Baumgardt (2021)")
        for coord_x in tqdm(cadidateStarCoords):
            coord_x_deg = [coord_x.ra.deg, coord_x.dec.deg]
            # Query the KD-tree for nearby coordinates
            nearby_indices = self.baumgardtTree.query_ball_point(coord_x_deg, radius.to_value(u.deg))
            matched_indices.append(nearby_indices)

        probabilityList = []
        gMagnitudeList = []
        for ind in matched_indices:
            probabilityList.append(self.baumgardtData[16][ind].values)
            gMagnitudeList.append(self.baumgardtData[12][ind].values)

        gMagnitudeList = np.array(gMagnitudeList)
        # breakpoint()
        gMagDiff = []
        for i in range(len(gMagnitudeList)):
            if len(gMagnitudeList[i]) > 0:
                gMagDiff.append(abs(self.g[i] - gMagnitudeList[i]))
            else:
                gMagDiff.append(-9.99)
        baumgardtChecklist = []
        for i in range(len(probabilityList)):
            if len(probabilityList[i]) > 0 and max(probabilityList[i]) > 0.5 > min(np.array(gMagDiff)[i]):
                baumgardtChecklist.append(1)
            else:
                baumgardtChecklist.append(0)

        return np.array(baumgardtChecklist)
