"""
Script to import GPR time slice and process with three different versions of
the Hough transform

Andrew Pretorius     20/05/20
"""

from matplotlib import cm
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time

start = time.time()

""" Import and processing """
# import image of time slice
img = cv2.imread(r'C:\Users\andre\Documents\uni\SOEE3058_Independent_Project\data\hough_test\slice5_env.png',0)

# apply gaussian blur
blur = cv2.GaussianBlur(img,(15,15),0)

# apply threshold
ret,th1 = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)


""" Apply Hough transforms """
# standard Hough transform
h, theta, d = hough_line(th1)

# probabilistic Hough transform
lines = probabilistic_hough_line(th2,line_length=300,line_gap=10)

# circlular Hough transform
imgt = cv2.GaussianBlur(img,(7,7),0)
ret,imgt = cv2.threshold(imgt,130,255,cv2.THRESH_BINARY)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(imgt,cv2.HOUGH_GRADIENT,1,1000,param1=50,param2=50,minRadius=0,maxRadius=0)

""" display results """
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=cm.gray)
row1, col1 = img.shape
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=100, min_angle=100, threshold=None)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
    ax[0].plot((0, col1), (y0, y1), '-r')
ax[0].axis((0, col1, row1, 0))
ax[0].set_title('(A) Standard Hough transform results')
ax[0].set_ylabel('Pixels')
ax[0].set_xlabel('Pixels')

ax[1].imshow(img, cmap='gray')
for line in lines:
    p0, p1 = line
    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[1].set_xlim((0, img.shape[1]))
ax[1].set_ylim((img.shape[0], 0))
ax[1].set_title('(B) Probabilistic Hough transform results')
ax[1].set_ylabel('Pixels')
ax[1].set_xlabel('Pixels')

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
ax[2].imshow(cimg, cmap='gray')
ax[2].set_xlim((0, img.shape[1]))
ax[2].set_ylim((img.shape[0], 0))
ax[2].set_title('(C) Circular Hough transform results')
ax[2].set_ylabel('Pixels')
ax[2].set_xlabel('Pixels')

plt.tight_layout()
plt.show()

end = time.time()
print(end - start)
