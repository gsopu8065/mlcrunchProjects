import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import cv2
import numpy as np

image = cv2.imread('./darkimage.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])
# plot the histogram
plt.figure()
plt.title("Before equalizeHist")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
x = np.arange(256, dtype='uint8')
y = [int(y[0]) for y in hist]
plt.bar(x, y)
plt.xlim([0, 256])

contractedImage = cv2.equalizeHist(image)
cv2.imshow("Contracted Image", contractedImage)

hist = cv2.calcHist([contractedImage], [0], None, [256], [0, 256])
# plot the histogram
plt.figure()
plt.title("After equalizeHist")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
x = np.arange(256, dtype='uint8')
y = [int(y[0]) for y in hist]
plt.bar(x, y)
plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()