#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 10:52:27 2022

@author: natthanan
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns


hdulist= fits.open('A1_mosaic.fits')
headers = hdulist[0].header
data = hdulist[0].data

from findpeaks import findpeaks

X=data.copy()[300:1000,300:1000]

fig, (ax1, ax2) = plt.subplots(1,2)


X1=X.copy()

#X_flat = X1.flatten()
#X0 = [0 if (d<3481).all() else d for d in X_flat]
#X = np.reshape(X0, np.shape(X1))

obj_idxs = []
for j,row in enumerate(X1):
    for i,pixval in enumerate(row):
        if pixval < 3481:
            X1[j][i] = 0
        else:
            obj_idxs.append([i,j])
obj_idxs = np.array(obj_idxs)

ax1.imshow(X)
ax2.imshow(X1)
plt.show()


#%%

import scipy.cluster.hierarchy as hcluster
import seaborn as sns
from itertools import cycle
thresh = 5
clusters = hcluster.fclusterdata(obj_idxs, thresh, criterion="distance")


# plotting
sns.scatterplot(*np.transpose(obj_idxs), hue=clusters, palette='Paired', s=5, legend=False)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()






#%%

#IRRELEVANT CODES


from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.colors as mcolors


#centers, indices = kmeans_plusplus(obj_idxs, n_clusters=7, random_state=0)

kmeans = KMeans(n_clusters=300)
kmeans.fit(obj_idxs)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

colors = 'tab20b'#mcolors.XKCD_COLORS#["#4EACC5", "#FF9C34", "#4E9A06", "m","red","yellow","blue"]

for k, col in enumerate(colors):
    cluster_data = labels == k
    plt.scatter(obj_idxs[cluster_data, 0], obj_idxs[cluster_data, 1], cmap=col, marker=".", s=10)

plt.scatter(centers[:, 0], centers[:, 1], c="black", s=10)
plt.title("K-Means++ Initialization")
plt.xticks([])
plt.yticks([])
plt.show()
#labels = kmeans.labels_
#cluster_centers = kmeans.cluster_centers_

#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)
#%%
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X1[my_members, 0], X1[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

#%%

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=7, n_features=2,random_state=0)


#%%
fp = findpeaks(method='mask')
# Fit
fp.fit(X1)

# Plot the pre-processing steps
fp.plot_preprocessing()
# Plot all
fp.plot()

# Initialize
fp = findpeaks(method='topology')
# Fit
fp.fit(X1)
