import cv2

faceCascadeDetector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
eyesCascadeDetector = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")
image = cv2.imread("./testimages/test2.jpg")

#step1: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#step2: detect faces
faces = faceCascadeDetector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

#step3: draw each face on the image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    gray_face = gray[y:y + h, x:x + w]  #extracting face from grayscale
    color_face = image[y:y + h, x:x + w] #extracting face from color

    # step4: detect eyes on each face
    eyes = eyesCascadeDetector.detectMultiScale(gray_face,
                                                scaleFactor=1.05)

    # step5: draw eyes on each face
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(color_face, (ex, ey), (ex + ew, ey + eh),
                      (0, 0, 255), 2)

#step6: display the image
cv2.imshow("Image", image)
cv2.waitKey(0)