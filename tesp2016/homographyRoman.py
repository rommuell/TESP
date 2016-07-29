# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:43:04 2016

@author: ROMANMUELLER
"""
import cv2 
import numpy as np


img0_path = "./photo.png"
img1_path = "./original.png"
img0 = cv2.imread(img0_path,0)
img1 = cv2.imread(img1_path,0)

#%% manual matching
src_pts = np.array([[0, 0],[639, 0], [639, 479], [0, 479]])
dst_pts = np.array([[185, 34],[521, 43],[497, 458],[185, 319]])

#%% automatic feature matching
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img0,None)
kp2, des2 = sift.detectAndCompute(img1,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
MIN_MATCH_COUNT = 4        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#%%

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

img2 = cv2.warpPerspective(img1, H, (img1.shape[1],img1.shape[0]))
img3 = cv2.warpPerspective(img0, np.linalg.inv(H), (img1.shape[1],img1.shape[0]))

cv2.imshow("origninal", img0)
cv2.imshow("photo", img1)
cv2.imshow("transform", img2)
cv2.imshow("transfom2", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()





