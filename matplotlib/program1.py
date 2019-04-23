from matplotlib import pyplot as plt
import numpy as np

def simple():
    plt.plot([1,2,3], [1,2,9])
    plt.show()

def multipleLine():
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()

def scatterPlot():
    plt.scatter([1,2,3], [1,2,9])
    plt.show()

def barPlot():
    plt.bar([1,2,3], [1,2,9])
    plt.show()

barPlot()
#multipleLine()
#scatterPlot()