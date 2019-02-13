import cv2
import numpy as np

image = cv2.imread('./sudoku.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)

def Eroded():
    for i in (1,2):
        eroded = cv2.erode(image.copy(), None, iterations=i)
        cv2.imshow("Eroded {} times".format(i), eroded)

def Dilation():
    for i in (1,2):
        eroded = cv2.dilate(image.copy(), None, iterations=i)
        cv2.imshow("Eroded {} times".format(i), eroded)

def Opening():
    kernel = np.ones((3,3), dtype="uint8") # or kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(image.copy(), cv2.MORPH_OPEN, kernel)
    cv2.imshow("opened", opened)

def Closing():
    kernel = np.ones((3,3), dtype="uint8") # or kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(image.copy(), cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closed", closed)

def Gradient():
    kernel = np.ones((3,3), dtype="uint8") # or kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(image.copy(), cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("closed", closed)

def WhiteHat():
    kernel = np.ones((3,3), dtype="uint8") # or kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(image.copy(), cv2.MORPH_TOPHAT, kernel)
    cv2.imshow("closed", closed)


Eroded()

cv2.waitKey(0)
cv2.destroyAllWindows()