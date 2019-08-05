from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt

#step1: load dataset
digitDataset = datasets.load_digits()
print("Labels: ", digitDataset.target_names)
print(digitDataset.data.shape)

#step2: split training data and test data
# 70% training and 30% test
(trainData, testData, trainLabels, testLabels) = train_test_split(digitDataset.data, digitDataset.target, test_size=0.3,random_state=109)
print("Total data: {}".format(digitDataset.data.shape[0]))
print("Training data: {}".format(len(trainData)))
print("Test data: {}".format(len(testData)))

#step3: create a model and train the data
model = KNeighborsClassifier(n_neighbors=10)
model.fit(trainData, trainLabels)

#step4: test the model to predict testdata
predictions = model.predict(testData)
print("Display Test report")
print(metrics.classification_report(testLabels, predictions))
print("Accuracy:",metrics.accuracy_score(testLabels, predictions))


#step5: display few digits and see the results
for i in range(6):
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]
	image = image.reshape((8, 8)).astype("uint8")

	plt.subplot(2, 4, i + 1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Digit: '+str(prediction))

plt.show()
