import cv2
import numpy as np

image = cv2.imread('./signs.png')
cv2.imshow("Original Sign", image)

#convert to grayscala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur the image
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#thresholding
(thresholdValue, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", thresh)

def task1():
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    cv2.imshow("All EXTERNAL Contours", image)


def task2():
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    cv2.imshow("All LIST Contours", image)

task1()
task2()
cv2.waitKey(0)
cv2.destroyAllWindows()