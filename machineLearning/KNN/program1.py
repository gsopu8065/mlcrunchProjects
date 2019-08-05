from sklearn.neighbors import KNeighborsClassifier

#Training data and labels
data = [[100], [200], [300], [700], [800]]
labels = [1, 1, 1, 2, 2]

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(data, labels)

#600 is close neighbors of 700 and 800 which has label(2)
print("First Prediction:")
print(neigh.predict([[600]]))
print(neigh.predict_proba([[600]]))

#250 is close neighbors of 100 and 200 which has label(1)
print("\nSecond Prediction:")
print(neigh.predict([[250]]))
print(neigh.predict_proba([[250]]))

#500 is right in the middle group1 (100,200,300) and
# group2(700, 800)
print("\nThird Prediction:")
print(neigh.predict([[500]]))
print(neigh.predict_proba([[500]]))


