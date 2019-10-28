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


namelabels = ["German_shepherd", "poodle", "hunting_dog"]
# Step1:  prepare training data
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


flowers, labels = prepareTrainingData("./dogs")
print("Total dogs: ", len(flowers))
print("Total labels: ", len(labels))

# step2: split training and test data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(flowers), np.array(labels), test_size=0.1,
                                                                  random_state=59)

print("Training data: {}".format(len(trainData)))
print("Training Labels: {}".format(len(trainLabels)))

print("Test data: {}".format(len(testData)))
print("Test labels: {}".format(len(testLabels)))

model = DecisionTreeClassifier(random_state=79)
#model = RandomForestClassifier(n_estimators=30, random_state=59)

# Train the model using the training sets
model.fit(trainData, trainLabels)

# Predict the response for test dataset
predictions = model.predict(testData)

print("Display Test report")
print(metrics.classification_report(testLabels, predictions))
print("Accuracy:", metrics.accuracy_score(testLabels, predictions))


# step5: display few digits and see the results
def testImage(imagePath):
    image = cv2.imread(imagePath)
    prediction = model.predict(getFeatures(image).reshape(1, -1))[0]
    cv2.putText(image, namelabels[prediction], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow("Test", image)
    cv2.waitKey(0)


testImage('./dogs/German_shepherd/n02106662_104.jpg')
testImage('./dogs/hunting_dog/n02116738_662.jpg')
testImage('./dogs/poodle/n02113624_35.jpg')

