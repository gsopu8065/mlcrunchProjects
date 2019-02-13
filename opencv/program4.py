import cv2
import numpy as np

image = cv2.imread("./bill.png")
rows,cols,c = image.shape

for i in range(10):
    M = np.float32([[1, 0, -10*i], [0, 1, 10*i]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imshow("image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()