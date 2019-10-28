from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn import metrics

# function to convert image to features
def getFeatures(image):
    image = cv2.resize(image, (500, 500))
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

# Step1:  prepare training data
def prepareTrainingData(folderPath):

    allImages = []
    allImagePaths = []

    # Read each image
    for imageName in np.array(os.listdir(folderPath)):

        if imageName.startswith("."):
            continue

        # read each image
        image = cv2.imread(folderPath + "/" + imageName)
        allImages.append(getFeatures(image))
        allImagePaths.append(image)

    return allImages, allImagePaths

allImages, allImagePaths = prepareTrainingData("./images")
print("Image Count: ", len(allImages))

clt = KMeans(n_clusters=3)
labels = clt.fit_predict(allImages)
print("Total labels = ",len(labels))
print("Labels are ",labels)

# Count labels
countLabels = {}
#for (a, index) in labels:
for index, a in enumerate(labels):
    if a in countLabels.keys():
        countLabels[a] += 1
    else:
        countLabels[a] = 1

    cv2.imshow("Cluster {0}, Image # {1}".format(a,countLabels[a]), cv2.resize(allImagePaths[index], (500, 500)))

# display cluster count
for key, value in countLabels.items():
    print("Cluster {0} has {1} images ".format(key, value))

cv2.waitKey(0)