from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
hdulist= fits.open('./A1_mosaic.fits')
headers = hdulist[0].header
data = hdulist[0].data
data = data.flatten()
data_filter = [d for d in data if (d>300 and d<4000).all() ]
plt.hist(data_filter, bins=200)
#plt.yscale('log')
plt.show()
