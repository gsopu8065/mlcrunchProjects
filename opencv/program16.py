import cv2
import numpy as np


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)


image = cv2.imread('./bill.png')
kernelSizes = [(3, 3), (9, 9), (7, 7)]
cv2.imshow('Original', image)

def blur1():
    for (kX, kY) in kernelSizes:
        blurKernal = np.ones((kX, kY), np.float32) / (kX*kY)
        blurred = convolve(image, blurKernal)
        cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

def blur2():
    for (kX, kY) in kernelSizes:
        blurred = cv2.blur(image, (kX, kY))
        cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

def gaussian():
    for (kX, kY) in kernelSizes:
        blurred = cv2.GaussianBlur(image, (kX, kY), 0)
        cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

def medianBlur():
    for k in (3, 9, 15):
        blurred = cv2.medianBlur(image, k)
        cv2.imshow("Median {}".format(k), blurred)
    cv2.waitKey(0)

def bilateral():
    image = cv2.imread('./lake.png')
    cv2.imshow('Original', image)
    blurred = cv2.bilateralFilter(image, 11, 61, 39)
    cv2.imshow("Blurred bilateral", blurred)
    cv2.waitKey(0)

bilateral()