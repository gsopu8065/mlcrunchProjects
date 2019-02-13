import cv2
import numpy as np

image = cv2.imread("./bill.png")
rows,cols = image.shape[:2]

for i in range(4):
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), -90, 1.0)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imshow("image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()