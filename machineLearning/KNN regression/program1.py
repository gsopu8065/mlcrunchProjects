import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

#step1: Load the data into dataframe
df_knn = pd.read_csv("airbnb_data.csv", low_memory=False)
print(df_knn[0:5])

#step2: Split the data into training (80%) and testing (20%)
(trainData, testData, trainLabels, testLabels) = train_test_split(df_knn[['accommodates','bathrooms','bedrooms','beds']],df_knn['price'], test_size=0.2, random_state = 42)
print("Training data: {}".format(len(trainData)))
print("Test data: {}".format(len(testData)))
print("Total data: {}".format(len(df_knn)))

#step3: Create a model and train it with the training data
model = KNeighborsRegressor(n_neighbors=7)
model.fit(trainData, trainLabels)

#step4: Use the trained model for prediction
predictions = model.predict(testData)
predictions = pd.DataFrame(predictions)

#step5: Analysis the predictions
print('Mean Absolute Error: $%.2f' % mean_absolute_error(testLabels, predictions))
print("Root Mean Squared Error: " + str(round(sqrt(mean_squared_error(testLabels, predictions)), 2)))

print("Price of a house accommodates 5, 2 bathrooms, 3 bedrooms and 3 beds: $" +str(model.predict([[5,2,3,3]])[0]))
print("Price of a house accommodates 3, 1 bathrooms, 1 bedrooms and 2 beds: $" +str(model.predict([[3,1,1,2]])[0]))