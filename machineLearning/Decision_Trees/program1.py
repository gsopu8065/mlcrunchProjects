from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import mahotas
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# function to convert image to features
# color and texture
def getFeatures(image):
    image = cv2.resize(image, (200, 200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #shape descriptor
    huFeature = cv2.HuMoments(cv2.moments(gray)).flatten()

    #color descriptor
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    # texture descriptor
    haralick = mahotas.features.haralick(gray).mean(axis=0)

    #combind color and texture
    return np.hstack([huFeature, hist.flatten(), haralick])

# Step1:  prepare training data
namelabels = ["apple", "avocado", "banana"]
def prepareTrainingData(folderPath):
    flowers = []
    labels = []

    # Read each directory
    for directoryName in os.listdir(folderPath):

        if directoryName.startswith(".") or directoryName not in namelabels:
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

flowers, labels = prepareTrainingData("./fruits")
print("Total flowers: ", len(flowers))
print("Total labels: ", len(labels))

# step2: split training and test data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(flowers), np.array(labels), test_size=0.2,
                                                                  random_state=59)

print("Training data: {}".format(len(trainData)))
print("Training Labels: {}".format(len(trainLabels)))

print("Test data: {}".format(len(testData)))
print("Test labels: {}".format(len(testLabels)))

model = DecisionTreeClassifier(random_state=77)
#model = RandomForestClassifier(n_estimators=30, random_state=89)

# Train the model using the training sets
model.fit(trainData, trainLabels)

# Test the model with testing dataset
predictions = model.predict(testData)

print("Display Testing report")
print(metrics.classification_report(testLabels, predictions))
print("Accuracy:", metrics.accuracy_score(testLabels, predictions))


# step5: display few digits and see the results
def testImage(imagePath):
    image = cv2.imread(imagePath)
    prediction = model.predict(getFeatures(image).reshape(1, -1))[0]
    cv2.putText(image, namelabels[prediction], (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    image = cv2.resize(image, (500, 500))
    cv2.imshow("Test", image)
    cv2.waitKey(0)


testImage('./fruits/apple/apple11.png')
testImage('./fruits/avocado/avocado11.jpg')
testImage('./fruits/banana/banana11.jpg')

