from matplotlib import pyplot as plt
import numpy as np

def axisLimit():
    plt.bar([1,2,3], [1,2,9])
    plt.xlim(0,100)
    plt.ylim(0,50)
    plt.show()

def titles():
    plt.bar([1, 2, 3], [1, 2, 9])
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    plt.title("Testing")
    plt.xlabel("Test X-label")
    plt.ylabel("Test Y-label")
    plt.show()

def createsFigure():
    plt.figure()
    x = np.arange(255, dtype='uint8')
    y = np.random.randint(0,255, size=255, dtype="uint8")
    plt.bar(x, y)

    #plt.figure()
    #plt.plot([1, 2, 3], [1, 4, 9], "r--")
    plt.show()


createsFigure()