import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import astropy.units as u
from tqdm import tqdm

class Comparer:
    def __init__(self, clusterName, candStarCoordinates):
        self.clusterName = clusterName
        self.baumgardtData = pd.read_csv(f"BaumgardtTables/{self.clusterName}.txt", sep="\t",
                                         header=None)
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
        for ind in matched_indices:
            probabilityList.append(self.baumgardtData[16][ind])

        baumgardtChecklist = []
        for i in probabilityList:
            if len(i) > 0 and max(i) > 0.5:
                baumgardtChecklist.append(1)
            else:
                baumgardtChecklist.append(0)

        return np.array(baumgardtChecklist)

