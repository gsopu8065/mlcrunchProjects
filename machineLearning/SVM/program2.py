from sklearn.model_selection import train_test_split
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


namelabels = ["daisy", "fritillary", "sunflower", "tigerlily"]
# Step1:  prepare training data
def prepareTrainingData(folderPath):
    flowers = []
    labels = []

    # Read each directory
    for directoryName in os.listdir(folderPath):

        if directoryName.startswith("."):
            continue

        # Read each image
        for imageName in os.listdir(folderPath + "/" + directoryName):

            if imageName.startswith("."):
                continue

            # read each image
            imagePath = folderPath + "/" + directoryName + "/" + imageName
            image = cv2.imread(imagePath)

            flowers.append(getFeatures(image))
            labels.append(namelabels.index(directoryName))

    return flowers, labels


flowers, labels = prepareTrainingData("./flowers")
print("Total flowers: ", len(flowers))
print("Total labels: ", len(labels))

# step2: split training and test data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(flowers), np.array(labels), test_size=0.1,
                                                                  random_state=59)

print("Training data: {}".format(len(trainData)))
print("Training Labels: {}".format(len(trainLabels)))

print("Test data: {}".format(len(testData)))
print("Test labels: {}".format(len(testLabels)))

clf = svm.SVC(C=100)

# Train the model using the training sets
clf.fit(trainData, trainLabels)

# Predict the response for test dataset
predictions = clf.predict(testData)

print("Display Test report")
print(metrics.classification_report(testLabels, predictions))
print("Accuracy:", metrics.accuracy_score(testLabels, predictions))


# step5: display few digits and see the results
def testImage(imagePath):
    image = cv2.imread(imagePath)
    prediction = clf.predict(getFeatures(image).reshape(1, -1))[0]
    cv2.putText(image, namelabels[prediction], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imshow("Test", image)
    cv2.waitKey(0)


testImage('./flowers/daisy/image_0801.jpg')
testImage('./flowers/sunflower/image_0721.jpg')
testImage('./flowers/fritillary/image_0644.jpg')
testImage('./flowers/tigerlily/image_0484.jpg')
