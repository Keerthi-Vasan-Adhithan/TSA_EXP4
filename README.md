### Developed By: Keerthi Vasan A
### Register No: 212222240048
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1) Import necessary libraries.
2) Load the dataset, convert the 'Date' column to datetime, and set it as the index. Plot the 'Adj Close' column.
3) Perform the ADF test on the 'Adj Close' data to check stationarity.
4) If the data is not stationary, apply differencing and recheck stationarity using the ADF test.
5) Plot the differenced data to visualize the stationary series.
6) Fit the ARMA(1,1) model on the differenced data and print the model summary.
7) Generate future predictions for 200 points using the fitted model.
8) Plot the original differenced series and predicted values on the same graph.
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load the dataset
df = pd.read_csv('/content/apple_stock.csv')

# Convert 'Date' column to datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use the 'Adj Close' column for time series analysis
df['Adj Close'].plot(figsize=(10, 6), title='Adjusted Close Price Over Time')
plt.ylabel('Adjusted Close Price (USD)')
plt.grid(True)
plt.show()

# Step 1: Check for stationarity using Augmented Dickey-Fuller (ADF) test
def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')

adf_test(df['Adj Close'])

# Step 2: If not stationary, difference the data to make it stationary
df_diff = df['Adj Close'].diff().dropna()

# Check for stationarity again after differencing
adf_test(df_diff)

# Step 3: Plot the differenced data
plt.figure(figsize=(10, 6))
plt.plot(df_diff, color='blue')
plt.title('Differenced Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Differenced Price (USD)')
plt.grid(True)
plt.show()

# Step 4: Fit the ARMA(1, 1) model on the differenced data
# The order (p, d, q) for ARIMA is (1, 0, 1) since we already differenced the data
model = ARIMA(df_diff, order=(1, 0, 1))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Step 5: Make predictions with the ARMA model
start = len(df_diff)
end = start + 200  # Predicting 200 points ahead
predictions = model_fit.predict(start=start, end=end)

# Step 6: Plot the differenced data and the model's predictions
plt.figure(figsize=(10, 6))
plt.plot(df_diff, label='Differenced Original Series', color='blue')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.title('ARMA Model Predictions on Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Price (USD)')
plt.grid(True)
plt.show()

```

### OUTPUT:
 1. Dataset:

![Screenshot 2024-09-25 091607](https://github.com/user-attachments/assets/734a7de7-989f-4dd7-ae73-1500d1e596ce)

 2. ADF and P values:

![Screenshot 2024-09-25 091638](https://github.com/user-attachments/assets/30f15f59-9383-46de-bb95-ad22ba2e6220)

 3. Model summary:
    
![image](https://github.com/user-attachments/assets/737c6172-b5c0-44ea-b7eb-cd48a309bb75)

  4. ARMA Model:

![Screenshot 2024-09-25 091947](https://github.com/user-attachments/assets/59ab5c84-f2bc-4210-9ea4-b248a22e48fc)




### RESULT:
Thus, a python program is successfully created to fit ARMA Model.
