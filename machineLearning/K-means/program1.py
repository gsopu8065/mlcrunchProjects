import cv2
from sklearn.cluster import KMeans
import numpy as np

#Step1: Read image
source = cv2.imread("./dog.jpg")
print("Image shape = ", source.shape)

#Step2:  Reshape matrix to a big array
image = source.reshape((source.shape[0] * source.shape[1], 3))
print("Big array shape = ", image.shape)

#Step3: Create KMeans model and train with an image
clt = KMeans(n_clusters=3)
clt.fit(image)
print("Total pixels = ",len(clt.labels_))

#Step4: count pixels in each cluster
countLabels = {}
for a in clt.labels_:
    if a in countLabels.keys():
        countLabels[a] += 1
    else:
        countLabels[a] = 1

# display cluster count
for key, value in countLabels.items():
    print("Cluster {0} has {1} pixels ".format(key, value))


# cluster center values
print("Cluster 0 color : ", clt.cluster_centers_[0].astype("uint8"))
print("Cluster 1 color : ", clt.cluster_centers_[1].astype("uint8"))
print("Cluster 2 color : ", clt.cluster_centers_[2].astype("uint8"))

color1 = np.zeros((20,20,3), dtype='uint8')
color1[:,:] = clt.cluster_centers_[0].astype("uint8")

color2 = np.zeros((20,20,3), dtype='uint8')
color2[:,:] = clt.cluster_centers_[1].astype("uint8")

color3 = np.zeros((20,20,3), dtype='uint8')
color3[:,:] = clt.cluster_centers_[2].astype("uint8")

#display results
cv2.imshow("Source Image", source)
cv2.imshow("Colors ", np.hstack((color1, color2, color3)))
cv2.waitKey(0)

