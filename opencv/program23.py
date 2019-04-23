import cv2
import numpy as np

image = cv2.imread('./star.png')
image = cv2.resize(image, (400,400))
cv2.imshow("Original Sign", image)


#convert to grayscala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur the image
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#thresholding
(thresholdValue, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", thresh)

#contours
contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]


def beforeApprox():
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def afterApprox():
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)


def pointsCount():
    peri = cv2.arcLength(contours[0], True)

    #before points counts
    print("Total points before approximation: {0} and perimeter: {1}".format(len(contours[0]), peri))

    #after contour approximated
    twentyPercentApprox = cv2.approxPolyDP(contours[0], 0.2 * peri, True)
    cv2.drawContours(image, twentyPercentApprox, -1, (255, 0, 0), 30)
    print("Twenty percent of arcLength: {0} and perimeter: {1}".format(len(twentyPercentApprox), 0.2 * peri))
    cv2.imshow("Twenty percent", image)
    cv2.waitKey(0)


    tenPercentApprox = cv2.approxPolyDP(contours[0], 0.1 * peri, True)
    cv2.drawContours(image, tenPercentApprox, -1, (0, 255, 0), 30)
    print("Ten percent of arcLength: {0} and perimeter: {1}".format(len(tenPercentApprox), 0.1 * peri))
    cv2.imshow("Ten percent", image)
    cv2.waitKey(0)

    onePercentApprox = cv2.approxPolyDP(contours[0], 0.01 * peri, True)
    cv2.drawContours(image, onePercentApprox, -1, (0, 0, 255), 30)
    print("One percent of arcLength: {0} and perimeter: {1}".format(len(onePercentApprox), 0.01 * peri))
    cv2.imshow("One percent", image)
    cv2.waitKey(0)

#beforeApprox()
#afterApprox()
#pointsCount()
#cv2.imshow("All Contours", image)


#def findShapes()





cv2.waitKey(0)
cv2.destroyAllWindows()