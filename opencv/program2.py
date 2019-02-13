import cv2
import numpy as np

image = np.random.randint(0,255, size=(150, 150, 3),dtype=np.uint8)
cv2.imshow('image', image)
cv2.imwrite('newimage.jpeg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()