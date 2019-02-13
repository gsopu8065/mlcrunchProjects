import cv2
import numpy as np

image = cv2.imread("./bill.png")
M = np.ones(image.shape, dtype='uint8') *100
resultImage = cv2.add(image, M)
cv2.imshow("Lighter Image", resultImage)




M = np.ones(image.shape, dtype='uint8') *50
resultImage = cv2.subtract(image, M)
cv2.imshow("brighter Image", resultImage)

cv2.waitKey(0)
cv2.destroyAllWindows()