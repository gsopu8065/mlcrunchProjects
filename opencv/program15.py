import cv2
import numpy as np

def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

#step1: read the input image
image = cv2.imread('./dogface.jpg', cv2.IMREAD_GRAYSCALE)

#step2: kernel to blur an image
kernel = np.array((
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
), dtype="int")

resultImage = cv2.filter2D(image, -1, kernel)

cv2.imshow('Original ', image)
cv2.imshow('Blur Image', blurredImage)

cv2.waitKey(0)
cv2.destroyAllWindows()


identity = np.array((
	[0, 0, 0],
	[0, 1, 0],
	[0, 0, 0]), dtype="int")
#print("Identity Kernal = {0}".format(identity))

edgeDetection = np.array((
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
), dtype="int")
#print("Edge detection Kernal = {0}".format(edgeDetection))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")
#print("Sharpen Kernal = {0}".format(sharpen))

#cv2.imshow('After Convolution', convolve(image, edgeDetection))




cv2.waitKey(0)
cv2.destroyAllWindows()