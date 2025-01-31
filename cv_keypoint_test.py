import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
queryImage = cv.imread('image_small.jpg', cv.IMREAD_COLOR)
trainImage = cv.imread('image.jpg', cv.IMREAD_COLOR)
 
# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
queryKeypoints, queryDescriptor = sift.detectAndCompute(queryImage, None)
trainKeypoints, trainDescriptor = sift.detectAndCompute(trainImage, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)

# filter matches using method from page 20 of:
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matches = flann.knnMatch(queryDescriptor, trainDescriptor, 2)
goodMatches : list[cv.DMatch] = []
for (cloest, next_closest) in matches:
    if cloest.distance < 0.45 * next_closest.distance:
        goodMatches.append(cloest)

outputImg = cv.drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, goodMatches, None)

cv.imwrite("output.png", outputImg)