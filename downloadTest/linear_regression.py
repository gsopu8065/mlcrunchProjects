import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#step1: load the training data
df = pd.read_csv('./SMSSpamCollection', delimiter='\t',header=None)
print("Training data ", df.shape)

#step2: split the training data in to two sets
vectorizer = TfidfVectorizer()
trainDataRaw, testDataRaw, trainLabels, testLabels = train_test_split(df[1],df[0])
trainData = vectorizer.fit_transform(trainDataRaw)
testData = vectorizer.transform(testDataRaw)
print("Training data: "+str(len(trainDataRaw)))
print("Test data: "+str(len(testDataRaw)))
print("Total data: "+str(len(df)))

#step3: create a model and train it
classifier = LogisticRegression()
classifier.fit(trainData, trainLabels)

#step4: Test the model with testing dataset
predictions = classifier.predict(testData)
print("Display Testing report")
print(metrics.classification_report(testLabels, predictions))
print("Accuracy:", metrics.accuracy_score(testLabels, predictions))


test1 = 'URGENT! Your Mobile No 1234 was awarded a Prize'
predictions = classifier.predict(vectorizer.transform([test1]))
print("Email with \""+test1+"\" is "+predictions[0])

test2 = 'Hey honey, whats up?'
predictions = classifier.predict(vectorizer.transform([test2]))
print("Email with \""+test2+"\" is "+predictions[0])
