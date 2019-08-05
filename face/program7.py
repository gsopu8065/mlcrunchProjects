import dlib
import cv2
import numpy as np

#step1: Loads face detection model
cnn_face_detector = dlib.cnn_face_detection_model_v1("./dnn/mmod_dog_hipsterizer.dat")

#step2: loads the image
image = cv2.imread("/Users/srujan.gopu/PycharmProjects/mlcrunchProjects/machineLearning/Linear_Regression/dogs/German_shepherd/n02106662_320.jpg")

#step3: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#step4: detect faces using CNN model
faces = cnn_face_detector(gray, 1)
for faceRect in faces:
    rect =  faceRect.rect
    '''x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # step5: draw rectangle around each face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    '''

    if(cv2.waitKey(1) != '\33'):
        x = rect.left() - 50
        y = rect.top() - 50
        w = rect.right() - x + 50
        h = rect.bottom() - y + 50
        face = image[y:y+h, x:x+w]
        cv2.imshow("face", face)

#step6: display the image
cv2.imshow("Image", image)
cv2.waitKey(0)