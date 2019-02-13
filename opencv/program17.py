import cv2
import numpy as np

def convolve(image, kernel):
    return cv2.filter2D(image, cv2.CV_64F, kernel)


image = cv2.imread('./sudoku.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', image)


def sobelX():
    sobelXKernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype="int8")
    sobelXImage = convolve(image, sobelXKernel)
    cv2.imshow("Sobel X Image", sobelXImage)
    cv2.waitKey(0)

def sobelX2():
    gX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0)
    cv2.imshow("Sobel X2 Image", gX)
    cv2.waitKey(0)

def sobelY():
    sobelXKernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]], dtype="int8")
    sobelXImage = convolve(image, sobelXKernel)
    cv2.imshow("Sobel Y Image", sobelXImage)
    cv2.waitKey(0)

def sobelY2():
    gX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1)
    cv2.imshow("Sobel Y2 Image", gX)
    cv2.waitKey(0)

def Laplacian():
    sobelXKernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]], dtype="int8")
    sobelXImage = convolve(image, sobelXKernel)
    cv2.imshow("Sobel Laplacian Image", sobelXImage)
    cv2.waitKey(0)

def Laplacian2():
    gX = cv2.Laplacian(image, ddepth=cv2.CV_64F)
    cv2.imshow("Sobel Laplacian2 Image", gX)
    cv2.waitKey(0)

sobelX()
sobelX2()
sobelY()
sobelY2()
Laplacian()
Laplacian2()