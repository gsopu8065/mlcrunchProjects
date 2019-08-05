from sklearn import svm

X = [[0, 0, 5], [1, 5, 1]]
y = [0, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

print(clf.predict([[-15, -15,5]]))

print(clf.support_vectors_)
print(clf.support_)