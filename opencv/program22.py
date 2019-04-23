import cv2
import numpy as np

#image = cv2.imread('./signs.png')
image = cv2.imread('./contourTest1.png')
cv2.imshow("Original Sign", image)

#convert to grayscala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur the image
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#thresholding
(thresholdValue, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", thresh)

#contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

def center():
    for c in contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 10, (0, 255, 0), -1)
    cv2.imshow("Center", image)

def area():
    for c in contours:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(image, "{:.2f}".format(area), (cX - 30, cY-60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 4)
    cv2.imshow("Area", image)

def perimeter():
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(image, "{:.2f}".format(perimeter), (cX - 30, cY-60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 4)
    cv2.imshow("Area", image)

def straightBoundingRect():
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Straight Bounding Rect", image)

def rotatedBoundingRect():
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.imshow("Rotated Bounding Rect", image)

def minimumClosingCircle():
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0, 255, 0), 2)
    cv2.imshow("Minimum closing circle", image)




#center()
#area()
#perimeter()
#straightBoundingRect()
rotatedBoundingRect()
#minimumClosingCircle()

cv2.waitKey(0)
cv2.destroyAllWindows()