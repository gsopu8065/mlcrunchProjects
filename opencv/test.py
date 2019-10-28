import cv2
import numpy as np

def myLUT(i, table):
    image = i.copy()
    rows, cols, c = image.shape
    for i in range(rows-1):
        for j in range(cols-1):
            image[i][j][0] = table[image[i][j][0]]
            image[i][j][1] = table[image[i][j][1]]
            image[i][j][2] = table[image[i][j][2]]
    return image


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(pow((i / 255.0),invGamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return np.hstack([cv2.LUT(image, table), myLUT(image, table)])

image = cv2.imread("./bill.png")

adjusted = adjust_gamma(image, gamma=2.2)

cv2.imshow("Images", np.hstack([image, adjusted]))
cv2.waitKey(0)

cv2.destroyAllWindows()