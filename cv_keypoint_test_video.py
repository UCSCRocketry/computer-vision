import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
#outputFourCC = cv.VideoWriter.fourcc(*'mp4v')
#outputVideo = cv.VideoWriter('output.mp4', outputFourCC, 20.0, (640,  480))

# Initiate SIFT detector
sift = cv.SIFT_create()

queryVideo = cv.VideoCapture("canal_drone.mp4")
trainImage = cv.imread('canal_satellite.png', cv.IMREAD_GRAYSCALE)
trainKeypoints, trainDescriptor = sift.detectAndCompute(trainImage, None)

while True:
    #print("Reading frame...")
    ret, queryImage = queryVideo.read()
    if not ret:
        break

    queryImage = cv.cvtColor(queryImage, cv.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    queryKeypoints, queryDescriptor = sift.detectAndCompute(queryImage, None)

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
        if cloest.distance < 0.7 * next_closest.distance:
            goodMatches.append(cloest)

    outputImg = cv.drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, goodMatches, None)

    cv.imshow("output", outputImg)

    cv.waitKey()

    # cv.imwrite("output.png", outputImg)
    #outputVideo.write(outputImg)

#print("Done.")
queryVideo.release()
#outputVideo.release()