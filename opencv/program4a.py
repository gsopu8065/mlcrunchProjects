import cv2
import numpy as np

image = cv2.imread("./bill.png")
#Use Image Width and Height for this simple example.
rows,cols,c = image.shape

cv2.imshow("image", image)
cv2.waitKey(0)
M = np.float32([[1, 0, -50], [0, 1, 40]])
image = cv2.warpAffine(image, M, (cols, rows))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
