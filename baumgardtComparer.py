import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

class Comparer:
    def __init__(self, clusterName, candStarCoordinates):
        self.clusterName = clusterName
        self.baumgardtData = pd.read_csv(f"BaumgardtTables/{self.clusterName}.txt", sep="\t",
                                         header=None)
        baumgardtRA = self.baumgardtData[1]
        baumgardtDec = self.baumgardtData[2]
        self.baumgardtCoords = list(zip(baumgardtRA, baumgardtDec))
        self.candidateList = candStarCoordinates

    def separationFinder(self):
        cadidateStarCoords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg) for ra, dec in self.candidateList]
        baumgardtCoords = [SkyCoord(ra=ra * u.deg, dec=dec * u.deg) for ra, dec in self.baumgardtCoords]

        radius = 3 * u.arcsec
        matched_indices = []
        print("-->Comparing memberships to Vasiliev & Baumgardt (2021)")
        for coord_x in tqdm(cadidateStarCoords):
            matched_indices_for_x = []
            for i, coord_b in enumerate(baumgardtCoords):
                separation = coord_x.separation(coord_b)
                if separation <= radius:
                    matched_indices_for_x.append(i)
            matched_indices.append(matched_indices_for_x)

        probabilityList = []
        for ind in matched_indices:
            probabilityList.append(self.baumgardtData[16][ind])

        baumgardtChecklist = []
        for i in probabilityList:
            if len(i) > 0 and min(i) > 0.5:
                baumgardtChecklist.append(1)
            else:
                baumgardtChecklist.append(1)

        return np.array(baumgardtChecklist)

