# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date:28-10-2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/blood_donor_dataset.csv')
data['created_at'] = pd.to_datetime(data['created_at'])

plt.plot(data['created_at'], data['pints_donated'])
plt.xlabel('Date')
plt.ylabel('Pints Donated')
plt.title('Pints Donated Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['pints_donated'])

plot_acf(data['pints_donated'])
plt.show()

plot_pacf(data['pints_donated'])
plt.show()

sarima_model = SARIMAX(data['pints_donated'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['pints_donated'][:train_size], data['pints_donated'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Pints Donated')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:
<img width="565" height="426" alt="image" src="https://github.com/user-attachments/assets/a87f58b7-0706-42eb-a8c1-2c32faabcb53" />

<img width="286" height="110" alt="image" src="https://github.com/user-attachments/assets/99823ae1-6dba-40de-b418-a914fa593daa" />

<img width="556" height="402" alt="image" src="https://github.com/user-attachments/assets/e4c143fa-d38e-45ff-bd3f-e770535f0171" />

<img width="580" height="430" alt="image" src="https://github.com/user-attachments/assets/f3637cfb-9a2c-4f19-ac33-564ea842fd75" />

<img width="612" height="420" alt="image" src="https://github.com/user-attachments/assets/03ef8bd7-1343-427a-8fb7-d3abad9519a8" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
