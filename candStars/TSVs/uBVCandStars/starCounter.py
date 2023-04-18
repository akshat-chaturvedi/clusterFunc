import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

for name in glob.glob('*_candStars_6.dat'):
    with open(r"%s"%name, 'r') as fp:
        lines = len(fp.readlines()) - 2
        print('Total Number of lines:', lines)


