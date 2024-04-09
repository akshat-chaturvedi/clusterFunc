import numpy as np
from astropy.table import Table
from datetime import datetime
import json

class GMagFinder:
    def __init__(self, clusterName, b, v, ra, dec):
        self.b = b
        self.v = v
        self.ra = ra
        self.dec = dec
        self.g = self.v - 0.0124*(self.b-self.v)
        gMagData = Table((self.g, self.ra, self.dec), names=("g", "ra", "dec"))
        gMagData.write(f"gData/{clusterName}.csv", format="ascii.csv", overwrite=True)
        print("---->G magnitudes file created")
