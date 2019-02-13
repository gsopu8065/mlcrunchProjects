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

#center()
#area()
#perimeter()
cv2.waitKey(0)
cv2.destroyAllWindows()