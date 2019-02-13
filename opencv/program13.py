import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import cv2
import numpy as np

def grayHistogram():
    image = cv2.imread('./bill.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", image)
    # construct a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # plot the histogram
    plt.figure()
    plt.title("Grayscale plot Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])

    # plot the histogram
    plt.figure()
    plt.title("Grayscale bar Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    x = np.arange(256, dtype='uint8')
    y = [int(y[0]) for y in hist]
    plt.bar(x, y)
    print("Max pixel count {0}".format(max(y)))
    print("Max color pixel is {0}".format(y.index(max(y))))
    plt.show()


def colorHistogram1():
    image = cv2.imread('./bill.png')
    cv2.imshow("Original", image)

    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color plot Histogram1")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

def histogramMasking():
    image = cv2.imread('./bill.png')
    cv2.imshow("Image", image)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (280, 310), (340, 360), 255, -1)
    cv2.imshow("Mask", mask)

    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'With out Mask Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])

    plt.figure()
    plt.title("'With Mask Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

#grayHistogram()
#colorHistogram1()
#histogramMasking()