#keras

#https://compvisionlab.wordpress.com/2013/04/07/convolution-opencv/
#https://en.wikipedia.org/wiki/Kernel_(image_processing)

import cv2
import numpy as np


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def myconvolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            tmp = (roi * kernel).sum()
            if tmp > 255:
                tmp = 255
            if tmp < 0:
                tmp = 0
            output[y - pad, x - pad] = tmp
    output = (output).astype("uint8")
    # for i in range(0, output.shape[0]):
    #     for j in range(0, output.shape[1]):
    #         signal_patch = image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
    #         tmp = (kernel * signal_patch).sum()
    #         # if tmp > 255:
    #         #     tmp = 255
    #         # if tmp < 0:
    #         #     tmp = 0
    #         output[i, j] = tmp
    return output

image = cv2.imread('./bill.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Before Convolution', image)

blurKernal = np.ones((5,5),np.float32)/25
print("Blur Kernal = {0}".format(blurKernal))

identity = np.array((
	[0, 0, 0],
	[0, 1, 0],
	[0, 0, 0]), dtype="int")
print("Identity Kernal = {0}".format(identity))

edgeDetection = np.array((
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
), dtype="int")
print("Edge detection Kernal = {0}".format(edgeDetection))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")
print("Sharpen Kernal = {0}".format(sharpen))

#cv2.imshow('After Convolution', myconvolve(image, edgeDetection))


x = np.array([[5,12,7,121],
              [146, 127, 220, 15],
              [144,129, 73,55],
              [56,29,56,74]], dtype="uint8")
y= np.array([[-1,-1,-1],
             [-1,8,-1],
             [-1,-1,-1]], dtype="uint8")

print(np.array_equal(myconvolve(image, edgeDetection), convolve(image, edgeDetection)))


cv2.waitKey(0)
cv2.destroyAllWindows()