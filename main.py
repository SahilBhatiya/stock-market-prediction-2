# Importing the required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv('stock_prices.csv')

# Creating the independent and dependent variables

# Independent variable
X = df['Day'].values.reshape(-1, 1)

# Dependent variables (Open, High, Low, Close, Volume, Adj Close)
y = df[['Day','Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].values

# Creating the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X, y)

# Predicting the values
y_pred = model.predict(X)

# Calculating the RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE: ", rmse)

# calculating the R2 score
r2 = model.score(X, y)
print("R2 score: ", r2)

# predicting the values for the next 30 days
next_30_days = pd.DataFrame({'Day': range(df['Day'].max() + 1, df['Day'].max() + 31)})
next_predictions = model.predict(next_30_days)

# Creating a new DataFrame to store the predictions
predictions_df = pd.DataFrame(next_predictions, columns=['Day', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
predictions_df['Day'] = predictions_df['Day'].astype(int)

# Saving the predictions to a new CSV file with a separate sheet
with pd.ExcelWriter('stock_predictions.xlsx') as writer:
    df.to_excel(writer, sheet_name='Original Data', index=False)
    predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
