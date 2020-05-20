"""
Script to import GPR data, estimate some attributes and run kmeans
to identify features

Andrew Pretorius     20/05/20
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import hilbert
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from copy import copy
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
import scipy
import time

start = time.time()

""" IMPORT DATA - This script requires a GPR grid in a single ASCII file"""
# file to be read:
f = np.loadtxt(r'C:\Users\andre\Documents\uni\SOEE3058_Independent_Project\data\Llanbedr_data\llanbedr_data.ASC')
# enter relevant paramters:
numberoflines = 1 
length = 2755
totaldepth = 379


# rearrange into grid (This step will require adjusting depending on the formatting of the ASCII file used)
A = np.zeros((numberoflines,totaldepth,length),dtype=float)
for i in range(numberoflines):
    for j in range(length):
        A[i][:,j] = f[i,(totaldepth*j):(totaldepth*(j+1))]
        
# cut off below 200 samples
depth = 200
lines = A[:,0:depth,:]

# delete leftover arrays to save memory
del f
del A


""" ATTRIBUTE ESTIMATION """

# envelope of signal using hilbert transform
envelope = np.zeros((numberoflines,depth,length),dtype=float)

for i in range(numberoflines):
    for j in range(length):
        envelope[i][:,j] = np.abs(hilbert(lines[i][:,j]))
    
env_smoothed = gaussian_filter(envelope, sigma=5)

# apply threshold to envelope
def threshold_AP(data, t_val):
    env_th = copy(data)
    env_th[env_th < t_val] = 0
    env_th[env_th > t_val] = 1
    return env_th

env_th7000 = threshold_AP(env_smoothed, 7000)

# rolling window standard deviation 
def sd_h(data, window):
    stdev = np.zeros((numberoflines,depth,length),dtype=float)
    window_width = window
    pad_width = np.int(window_width/2)
    for i in range(numberoflines):
        for j in range(depth):
            s = pd.Series(data[i][j,:])
            stdev[i][j,:] = s.rolling(window_width).std(center=True)
        stdev[i] = np.pad(stdev[i][:,pad_width:-pad_width], ((0,0),(pad_width,pad_width)), mode='edge')
    return stdev   

sd10 = gaussian_filter1d(sd_h(envelope, 10), sigma=10, axis=2)

# Calculate discontinuity attribute 
"""Functions sourced from https://github.com/seg/tutorials-2015/blob/master/1512_Semblance_coherence_and_discontinuity/writeup.md"""

def moving_window(data, func, window): 
    wrapped = lambda x: func(x.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)

def marfurt_semblance(region):#
# Stack traces in 3D region into 2D array
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape
    square_sums = (np.sum(region, axis=1))**2
    sum_squares = np.sum(region**2, axis=1)
    c = square_sums.sum() / sum_squares.sum()
    return c / ntraces

m_semblence = moving_window(lines, marfurt_semblance, (1, 9, 15))
m_semblence = gaussian_filter(m_semblence, sigma=10)



""" K-MEANS - Implementation of K-means is based on code sourced from https://github.com/James-Beckwith/hackathon_ABZ_18/ """  
# store original shape of data
originalShape = envelope[0].shape

# reshape each data type into vectors and store together in single array
a = np.reshape((env_smoothed[0]**2), [np.product(np.shape(lines[0])), 1])
b = np.reshape(env_th7000[0], [np.product(np.shape(lines[0])), 1])
c = np.reshape(sd10[0], [np.product(np.shape(lines[0])), 1])
d = np.reshape(m_semblence[0], [np.product(np.shape(lines[0])), 1])
input_data = np.hstack((a,b,c,d))

# convert NaN entries to zero
NaNs = np.isnan(input_data)
input_data[NaNs] = 0

# feature scaling
input_data = PowerTransformer(method='yeo-johnson').fit_transform(input_data)

# run kmeans
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=8).fit(input_data)

''' function to store clusters into an array of a given size. It is assumed that the array will be expanded across the first axis'''
def store_labels_original_size(labels, original_size, label_size):
    labels_vec = np.zeros(label_size)
    labels_vec = np.expand_dims(labels, axis=1)
    original_size_labels = np.reshape(labels_vec, original_size)
    return original_size_labels

# return original shape of data
clusts_reshape = store_labels_original_size(kmeans.labels_, originalShape, np.shape(input_data[:,0]))

# Display results

r = 400./1500. #aspect ratio
plt.figure(1)
plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
im1 = plt.imshow(clusts_reshape[:,0:1500], cmap=plt.cm.get_cmap('jet', 4), vmin=0, vmax=3, alpha=1.0, interpolation='nearest',extent=[0,60,40,0], aspect=r)
im2 = plt.imshow(lines[0][:,0:1500], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), alpha=0.5, interpolation='nearest',extent=[0,60,40,0], aspect=r)
plt.title('(B) K-means algorithm output')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')
cmap = plt.get_cmap("jet", 4)
norm= colors.BoundaryNorm(np.arange(0,5)-0.5, 4)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.arange(0,4), orientation='vertical', shrink=0.55, pad=0.015)
cbar.set_ticklabels(["Quiet", "Continuous horizontal reflections", "High amplitude reflections", "Chaotic reflections"])

plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
plt.imshow(lines[0][:,0:1500], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), interpolation='nearest',extent=[0,60,40,0], aspect=r)
plt.title('(A) Original data (Dataset A)')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')
cbar = plt.colorbar(orientation='vertical', shrink=0.55, pad=0.015)
cbar.set_ticks([(np.amax(lines[0]))*0.95, 0, (np.amin(lines[0]))*0.95,])
cbar.set_ticklabels(["Positive amplitude","0" ,"Negative amplitude"])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.2)


# small subplot 1
plt.figure(2)
plt.subplot2grid((2,2), (0,0))
plt.imshow(lines[0][75:175,576:826], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), interpolation='nearest',extent=[23,33,35,15], aspect=r)
plt.title('(A) Section of original data (Dataset A)')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')

# small subplot 2
plt.subplot2grid((2,2), (1,0))
im1 = plt.imshow(clusts_reshape[75:175,576:826], cmap=plt.cm.get_cmap('jet', 4), vmin=0, vmax=3, alpha=1.0, interpolation='nearest',extent=[23,33,35,15], aspect=r)
im2 = plt.imshow(lines[0][75:175,576:826], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), alpha=0.5, interpolation='nearest',extent=[23,33,35,15], aspect=r)
plt.title('(B) Clusters')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')

# small subplot 3
plt.subplot2grid((2,2), (0,1))
plt.imshow(lines[0][100:200,1051:1302], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), interpolation='nearest',extent=[42,52,40,20], aspect=r)
plt.title('(C) Section of original data (Dataset A)')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')
cbar = plt.colorbar(orientation='vertical', shrink=0.9, pad=0.015)
cbar.set_ticks([(np.amax(lines[0]))*0.95, 0, (np.amin(lines[0]))*0.95,])
cbar.set_ticklabels(["Positive amplitude","0" ,"Negative amplitude"])

# small subplot 4
plt.subplot2grid((2,2), (1,1))
im1 = plt.imshow(clusts_reshape[100:200,1051:1302], cmap=plt.cm.get_cmap('jet', 4), vmin=0, vmax=3, alpha=1.0, interpolation='nearest',extent=[42,52,40,20], aspect=r)
im2 = plt.imshow(lines[0][100:200,1051:1302], cmap='gray', vmin=np.amin(lines[0]), vmax=np.amax(lines[0]), alpha=0.5, interpolation='nearest',extent=[42,52,40,20], aspect=r)
plt.title('(D) Clusters')
plt.ylabel('Two-way time (ns)')
plt.xlabel('Distance (m)')
cmap = plt.get_cmap("jet", 4)
norm= colors.BoundaryNorm(np.arange(0,5)-0.5, 4)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=np.arange(0,4), orientation='vertical', shrink=0.9, pad=0.015)
cbar.set_ticklabels(["Quiet", "Continuous horizontal reflections", "High amplitude reflections", "Chaotic reflections"])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=-0.2)

end = time.time()
print(end - start)
