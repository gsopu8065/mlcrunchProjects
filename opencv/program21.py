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

    #start
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        print(len(approx))
        (x, y, w, h) = cv2.boundingRect(c)
        if len(approx) == 4: #for rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #end

    #cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
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