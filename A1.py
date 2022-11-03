from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns

hdulist= fits.open('A1_mosaic.fits')
headers = hdulist[0].header
data = hdulist[0].data
data_flat = data.flatten()
data_backg = [d for d in data_flat if (d>300 and d<4000).all()]
data_filter = [d for d in data_flat if (d>7000).all()]

#%%
plt.imshow(data,cmap='terrain')
plt.show()
#%%
plt.hist(data_backg, bins=1000)
plt.show()
#%%
plt.hist(data_filter, bins=200)
plt.show()
#%%

counts, edges, patches = plt.hist(data_backg, bins=1000)
counts_cut = [c for c in counts if (c<400000).all() & (c!=0).all()]
counts_cut_index = np.where((counts<400000) & (counts!=0))
#print(counts_cut_index)

centers = 0.5*(edges[1:]+ edges[:-1])
centers_cut = centers[counts_cut_index]
#plt.plot(centers_cut, counts_cut)
#plt.plot(centers,counts)

def gaussian(x, mu, sig,A):
    return A*np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2.)))

initial_guess = [3420,12,3e5]
po,po_cov = curve_fit(gaussian, centers_cut, counts_cut,initial_guess)
plt.plot(centers_cut, gaussian(centers_cut, po[0],po[1],po[2]))
plt.show()

print('Mean =  %.5e +/- %.5e' %(po[0],np.sqrt(po_cov[0,0])))
print('Sigma = %.3e +/- %.3e' %(po[1],np.sqrt(po_cov[1,1])))
print('A =  %.3e +/- %.3e' %(po[2],np.sqrt(po_cov[2,2])))

#%%
noise_mean = po[0]
noise_sigma = po[1]
obj_lower_bound = 5*noise_sigma + noise_mean
print(lower_bound)
plt.plot(centers_cut,counts_cut)
plt.plot(centers_cut, gaussian(centers_cut, po[0],po[1],po[2]))
plt.plot(lower_bound,0,'x')


#%%

data_clean = data

'''
data_clean[2218:2358,888:920] = 0
data_clean[3385:3442,2454:2478] = 0
data_clean[3198:3442,753:797] = 0
data_clean[1397:1454,2075:2102] = 0
data_clean[2698:2835,955:992] = 0
data_clean[2283:2337,2117:2147] = 0
data_clean[3700:3806,2117:2148] = 0
data_clean[4075:4117,547:576] = 0
'''

def mask(data,x1,x2,y1,y2,lowerbound=obj_lower_bound):
    for i in range(y2-y1):
        for j in range(x2-x1):
            if data[y1:y2,x1:x2][i][j] > obj_lower_bound:
                data[y1:y2,x1:x2][i][j] == 0 
                
                
                
    #data[y1:y2,x1:x2] = [0 for d in data[y1:y2,x1:x2] if (d>obj_lower_bound).all()]

mask(data_clean,888,920,2218,2358)
    


plt.imshow(data_clean)
plt.show()

plt.imshow(data)
plt.show()
