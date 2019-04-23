import cv2

haarCascadeDetector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
image = cv2.imread("./testimages/test1.jpg")

#step1: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#step2: detect faces
faces = haarCascadeDetector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

#step3: draw each face on the image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#step4: display the image
cv2.imshow("Image", image)
cv2.waitKey(0)