from stockai import Stock
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

td = Stock('TD.TO')
prices_list = td.get_historical_prices('2018-01-01', '2018-01-30')
stockPrices = prices_list.get('close')
dates = range(1, len(stockPrices) + 1)

dates = np.reshape(dates, (len(dates), 1)) # convert to 1xn dimension
stockPrices = np.reshape(stockPrices,(len(stockPrices), 1))

plt.plot(dates, stockPrices, "bo", dates, stockPrices)
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.title('Stock Price')

svrModel1  = SVR(kernel='linear', C=1e3)
svrModel2 = SVR(kernel='poly', C=1e3, degree=2)
svrModel3 = SVR(kernel='rbf', C=1e3, gamma=0.1)

svrModel1.fit(dates, stockPrices)
svrModel2.fit(dates, stockPrices)
svrModel3.fit(dates, stockPrices)

predictDates = [21, 22, 23, 24, 25]
predictDates = np.reshape(predictDates, (len(predictDates), 1))

predictedStockValueByLinear = svrModel1.predict(predictDates)
predictedStockValueByPoly = svrModel2.predict(predictDates)
predictedStockValueByRGF = svrModel3.predict(predictDates)

plt.plot(predictDates, predictedStockValueByLinear, "bo", predictDates, predictedStockValueByLinear, c='g')
plt.plot(predictDates, predictedStockValueByPoly, "bo", predictDates, predictedStockValueByPoly, c='c')
plt.plot(predictDates, predictedStockValueByRGF, "bo", predictDates, predictedStockValueByRGF, c='r')
plt.show()