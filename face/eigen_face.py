import cv2
import os
import numpy as np

namelabels = ["bill gates", "mark zuckerberg"]

#Code to detect face using harr cascade algorithm
haarCascadeDetector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
def detectFace(img):
    faces =  haarCascadeDetector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


#Step1:  prepare training data
def prepareTrainingData(folderPath):

    faces = []
    labels = []

    #Read each directory
    for directoryName in os.listdir(folderPath):

        if directoryName.startswith("."):
            continue

        #Read each image
        for imageName in os.listdir(folderPath + "/" + directoryName):

            if imageName.startswith("."):
                continue

            #read each image
            imagePath = folderPath + "/" + directoryName + "/" + imageName
            image = cv2.imread(imagePath)

            #convert to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #detect faces
            detectedFaces = detectFace(gray)

            #store face and lables in an array
            for (x, y, w, h) in detectedFaces:
                faces.append(gray[y:y + w, x:x + h])
                labels.append(namelabels.index(directoryName))

    return faces, labels


faces, labels = prepareTrainingData("./trainingImages")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#Step2:  create face recognizer and train the model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.train(faces, np.array(labels))

#step3: test the model by running test images and predict
def predictImage(img):

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect faces
    detectedFaces = detectFace(gray)

    for (x, y, w, h) in detectedFaces:

        #draw rectangle on each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #face model recognizer predicts the test image and returns the label
        label, confidence = faceRecognizer.predict(gray[y:y + w, x:x + h])

        #label is a number so get the associated name
        label_text = namelabels[label]

        #write name on the image
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 0), 2)

        #Show the image
        cv2.imshow("Predicted Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


testImage1 =  cv2.imread("./testingImages/test1.jpeg")
predictImage(testImage1)

testImage2 =  cv2.imread("./testingImages/test2.jpeg")
predictImage(testImage2)
