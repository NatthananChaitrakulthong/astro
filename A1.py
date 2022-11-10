from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns

hdulist= fits.open('A1_mosaic.fits')
headers = hdulist[0].header
data = hdulist[0].data
#%%
data_flat = data.flatten()
data_backg = [d for d in data_flat if (d>300 and d<4000).all()]
data_filter = [d for d in data_flat if (d>7000).all()]

#%%
plt.imshow(data)
plt.show()
#%%
#plt.hist(data_backg, bins=1000)
#plt.show()
#%%
#plt.hist(data_filter, bins=200)
#plt.show()
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
obj_lowerbound = 5*noise_sigma + noise_mean
artf_lowerbound = 6000

print(obj_lowerbound)
print(artf_lowerbound)


plt.plot(centers_cut,counts_cut)
plt.plot(centers_cut, gaussian(centers_cut, po[0],po[1],po[2]))
plt.plot(obj_lowerbound,0,'x')


#%%

data_clean = data.copy()

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

def mask(df,y1,y2,x1,x2,lowerbound=artf_lowerbound):
    artf_idx = []
    for i in range(y2-y1):
        for j in range(x2-x1):
            if df[y1:y2,x1:x2][i][j] > lowerbound:
                df[y1:y2,x1:x2][i][j] = 0 
                artf_idx.append([x1+j,y1+i])
    
    return artf_idx
                
                
#data[y1:y2,x1:x2] = [0 for d in data[y1:y2,x1:x2] if (d>obj_lower_bound).all()]

artf_idxs = []

artf_idxs.append(mask(data_clean,2218,2358,858,950,obj_lowerbound))
artf_idxs.append(mask(data_clean,888,920,2218,235))
artf_idxs.append(mask(data_clean,3385,3442,2434,2500,obj_lowerbound))
artf_idxs.append(mask(data_clean,3198,3442,728,835,obj_lowerbound))
artf_idxs.append(mask(data_clean,1397,1454,2050,2122,obj_lowerbound))
artf_idxs.append(mask(data_clean,2698,2835,920,1020,obj_lowerbound))
artf_idxs.append(mask(data_clean,2283,2337,2100,2160,obj_lowerbound))
artf_idxs.append(mask(data_clean,3700,3806,2100,2170,obj_lowerbound))
artf_idxs.append(mask(data_clean,4075,4117,530,596,obj_lowerbound))
artf_idxs.append(mask(data_clean,4320,4408,1100,1660,obj_lowerbound))
artf_idxs.append(mask(data_clean,557,597,1752,1790,obj_lowerbound))


#artf_idxs.append(mask(data_clean,0,4610,1015,1735))

artf_idxs.append(mask(data_clean,0,4610,1410,1457,obj_lowerbound)) #long rectangle for the giant star streak
artf_idxs.append(mask(data_clean,4010,4053,1410,1475,obj_lowerbound)) #long rectangle for the giant star streak
artf_idxs.append(mask(data_clean,2900,3500,1100,1800,obj_lowerbound))
artf_idxs.append(mask(data_clean,0,10,967,1720,obj_lowerbound))
artf_idxs.append(mask(data_clean,6,55,1628,1708,obj_lowerbound))
artf_idxs.append(mask(data_clean,10,25,1328,1505,obj_lowerbound))
artf_idxs.append(mask(data_clean,115,175,1290,1540,obj_lowerbound))
artf_idxs.append(mask(data_clean,210,320,1386,1482,obj_lowerbound))
artf_idxs.append(mask(data_clean,310,356,1010,1704,obj_lowerbound))
artf_idxs.append(mask(data_clean,422,457,1100,1653,obj_lowerbound))


#print(artf_idxs)
#%%
from matplotlib import colors

plt.imshow(data_clean)
plt.show()

#%%

hdulist= fits.open('A1_mosaic.fits')
headers = hdulist[0].header
data = hdulist[0].data

X=data_clean.copy()[-800:-200,200:-200]

fig, (ax1, ax2) = plt.subplots(1,2)


X1=X.copy()

#X_flat = X1.flatten()
#X0 = [0 if (d<3481).all() else d for d in X_flat]
#X = np.reshape(X0, np.shape(X1))

obj_idxs = []
for j,row in enumerate(X1):
    for i,pixval in enumerate(row):
        if pixval <= obj_lowerbound:
            X1[j][i] = 0
        else:
            obj_idxs.append([i,j])
obj_idxs = np.array(obj_idxs)

ax1.imshow(X)
ax2.imshow(X1)
plt.show()


#%%
 
import scipy.cluster.hierarchy as hcluster

thresh = 2.5
clusters = hcluster.fclusterdata(obj_idxs, thresh, criterion="distance")

#%%
# plotting
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X)
ax2.imshow(X1)
ax3.imshow(X1)
sns.scatterplot(*np.transpose(obj_idxs), hue=clusters, palette = 'Paired', s=5, legend=False, ax=ax3)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()

#%%

def galaxy_magnitude(cluster_labels, obj_indices):
    pixval_list = []
    pixval_sum = []
    pixval_max = []
    pixval_max_idxs = []
    for n in (set(cluster_labels)):
        pixvals = []
        ind = np.where(cluster_labels==n)
        pix_pos = obj_indices[ind]
        for pos in pix_pos:
            x = pos[0]
            y = pos[1]
            pixvals.append(X1[y][x])
        #if len(pixval_list) > 1:
        pixval_list.append(pixvals)
        pixval_sum.append(sum(pixvals))
        pixval_max.append(max(pixvals))
        pixval_max_idxs.append(pix_pos[np.where(pixvals == max(pixvals))])
    
    return pixval_list, pixval_sum, pixval_max, pixval_max_idxs

l, s, m, mi = galaxy_magnitude(clusters,obj_idxs)
print(s)

#%%

ZPinst  = 2.530E+01 
magi = [-2.5*np.log10(counts/ZPinst) for counts in s]

#m = ZPinst + np.array(magi)

f,axs = plt.subplots(1,2)

# histogram of counts of galaxies with different magnitude 
c, e, p = axs[0].hist(magi,bins=100)
cen = 0.5*(e[1:]+ e[:-1])
axs[0].set_ylabel('Counts')
axs[0].set_xlabel('m')

# Plot of N(m) against m
Mag_list = np.arange(min(magi),max(magi)+0.001,0.001)
Nm = [len([mag for mag in magi if mag<Mag]) for Mag in Mag_list]


axs[1].plot(Mag_list, np.log10(Nm))
axs[1].set_ylabel('logN(m)')
axs[1].set_xlabel('m')
plt.yscale('linear')
plt.show()
#y = np.log()

#%%
import pandas as pd 
d1 = {'Clluster no.':np.arange(1,max(clusters)+1) , 'Pixel Values' : l
     , 'Total Brightness (sum)': s, 'Brightness Pixel Value': m, 'Brightest Pixel Coordinate' : mi,'Magnitude': magi}

df1 = pd.DataFrame(data=d1)
display(df1)
df1.to_csv('Cluster_800200_200200.csv')

d2 = {'Magnitude (Step of 0.001)':Mag_list , 'N(m)' : Nm}
df2 = pd.DataFrame(data=d2)
display(df1)
df2.to_csv('Magnitude-Counts_800200_200200.csv')

#%%

#*******USELESS codes**********

data_galaxy = data.copy()

lst_all=[]
def find_galaxy(df):
    
    for j in range(2):#len(df[:,0]):
        lst_row=[]
        for pv,i in enumerate(data_galaxy[j]):
            while pv > obj_lowerbound:
                lst_row.append(i)
            
        print(lst_row)
    
find_galaxy(data_galaxy)

#