import glob

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join

from geotiff import GeoTiff


def read_tiff(filename):
    if not os.path.exists(filename):
        print('File not found: {}'.format(filename))
        return None
    geo_tiff = GeoTiff(filename)
    tiff_array = np.array(geo_tiff.read())
    return tiff_array