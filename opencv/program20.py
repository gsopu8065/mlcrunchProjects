import cv2
import numpy as np

image = cv2.imread('./sudoku.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)

def simple1():
    (thresholdValue, imageRes) = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", imageRes)

def simple2():
    (thresholdValue, imageRes) = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold inv", imageRes)

def otsu():
    (thresholdValue, threshInv) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("thresholdValue = {0}".format(thresholdValue))
    cv2.imshow("Threshold", threshInv)

def otsu2():
    (thresholdValue, threshInv) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold inv", threshInv)

def adaptiveMean():
    thresh = cv2.adaptiveThreshold(image, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 0)
    cv2.imshow("OpenCV Mean Thresh", thresh)

def adaptiveGaussian():
    thresh = cv2.adaptiveThreshold(image, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 0)
    cv2.imshow("OpenCV Gaussian Thresh", thresh)

adaptiveMean()
adaptiveGaussian()

cv2.waitKey(0)
cv2.destroyAllWindows()