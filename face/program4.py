import dlib
import cv2

#step1: converts to gray image
hogFaceDetector = dlib.get_frontal_face_detector()

#step1: converts to gray image
image = cv2.imread("./testimages/test4.jpg")

#step1: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#step1: converts to gray image
faces = hogFaceDetector(gray, 1)
for (i, rect) in enumerate(faces):
    # step1: converts to gray image
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # step1: converts to gray image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#step6: display the image
cv2.imshow("Image", image)
cv2.waitKey(0)