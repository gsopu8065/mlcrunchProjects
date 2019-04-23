from sklearn.neighbors import KNeighborsClassifier

data = [[0], [1], [200], [300], [600], [700]]
labels = [0, 0, 1, 1, 2, 2]

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(data, labels)

#750 is close neighbors of 600 and 700 which has label(2)
print("First Prediction:")
print(neigh.predict([[750]]))
print(neigh.predict_proba([[750]]))

#450 is close neighbors of 300 and 200 which has label(1)
print("\nSecond Prediction:")
print(neigh.predict([[450]]))
print(neigh.predict_proba([[450]]))


