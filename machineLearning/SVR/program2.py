from stockai import Stock
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


#step1: prepare the traing data
td = Stock('TD.TO')
prices_list = td.get_historical_prices('2018-01-01', '2018-01-30')
print(prices_list.keys())
stockPrices = prices_list.get('close')
dates = range(1, len(stockPrices) + 1)

dates = np.reshape(dates, (len(dates), 1)) # convert to 1xn dimension
stockPrices = np.reshape(stockPrices,(len(stockPrices), 1))
print("Date length "+str(len(dates)))
print("Stock price length "+str(len(stockPrices)))

#step2: Split the data into training (80%) and testing (20%)
(trainData, testData, trainLabels, testLabels) = train_test_split(dates,stockPrices, test_size=0.2, random_state = 42)
print("Training data: {}".format(len(trainData)))
print("Test data: {}".format(len(testData)))

#step3: Create a model and train it with the training data
svrModel = SVR(kernel='rbf', C=1e3, gamma=0.1)
svrModel.fit(trainData, trainLabels)

#step4: Use the trained model for prediction
predictions = svrModel.predict(testData)
predictions = pd.DataFrame({'price': predictions})

#step5: Analysis the predictions
print('Mean Absolute Error: $%.2f' % mean_absolute_error(testLabels, predictions))
print("Root Mean Squared Error: " + str(round(sqrt(mean_squared_error(predictions, testLabels)), 2)))

predictDates = [21, 22, 23]
predictDates = np.reshape(predictDates, (len(predictDates), 1))

predictedStockValue = svrModel.predict(predictDates)
print("Predicted Stock value on 21st day: "+ str(predictedStockValue[0]))
print("Predicted Stock value on 22st day: "+ str(predictedStockValue[1]))
print("Predicted Stock value on 23st day: "+ str(predictedStockValue[2]))