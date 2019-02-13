import cv2
import numpy as np

image = cv2.imread('./bill.png')


#cropping
face = np.copy(image[50:230, 100:270])

image[:,:] = [0,225,0]
image[50:230, 100:270] = face

cv2.imshow("bill", image)
cv2.imshow("bill face", face)

cv2.waitKey(0)
cv2.destroyAllWindows()