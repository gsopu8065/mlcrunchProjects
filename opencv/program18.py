import cv2
import numpy as np


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread('./bill.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Before Convolution', image)

def myAlgorithm():
    edgeDetection = np.array((
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ), dtype="int")

    cv2.imshow('After Convolution', convolve(image, edgeDetection))

def canny():
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 210)
    cv2.imshow('Edges', edges)


canny()


cv2.waitKey(0)
cv2.destroyAllWindows()