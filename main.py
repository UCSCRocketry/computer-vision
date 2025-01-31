import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import math;

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

trainCenterX = 0
trainCenterY = 0
queryCenterX = 0
queryCenterY = 0
for match in goodMatches:
    queryPointX, queryPointY = queryKeypoints[match.queryIdx].pt
    trainPointX, trainPointY = trainKeypoints[match.trainIdx].pt
    trainCenterX += trainPointX
    trainCenterY += trainPointY
    queryCenterX += queryPointX
    queryCenterY += queryPointY
trainCenterX /= len(goodMatches)
trainCenterY /= len(goodMatches)
queryCenterX /= len(goodMatches)
queryCenterY /= len(goodMatches)

angle = 0
for match in goodMatches:
    queryPointX, queryPointY = queryKeypoints[match.queryIdx].pt
    trainPointX, trainPointY = trainKeypoints[match.trainIdx].pt
    queryPointX -= queryCenterX
    queryPointY -= queryCenterY
    trainPointX -= trainCenterX
    trainPointY -= trainCenterY
    queryAngle = math.atan2(queryPointY, queryPointX)
    trainAngle = math.atan2(trainPointY, trainPointX)
    angle += queryAngle - trainAngle
angle /= len(goodMatches)
print("Angle: " + str(angle * (180 / math.pi)))

# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
 
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
p1 = (int(trainCenterX), int(trainCenterY))
p2 = (int(queryCenterX), int(queryCenterY))

outputImg = cv.copyTo(trainImage, None)

# outputImg = cv.drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, goodMatches, None)
# for match in goodMatches:
#     trainPointX, trainPointY = trainKeypoints[match.trainIdx].pt
#     queryPointX, queryPointY = queryKeypoints[match.queryIdx].pt
#     p1 = (int(trainPointX), int(trainPointY))
#     p2 = (int(queryPointX + translateX), int(queryPointY + translateY))
#     cv.line(outputImg, p1, p2, (255, 0, 0), 3)

cv.circle(outputImg, p1, 0, (0, 0, 255), 5)
cv.circle(outputImg, p2, 0, (255, 0, 0), 5)

cv.imwrite("output.png", outputImg)