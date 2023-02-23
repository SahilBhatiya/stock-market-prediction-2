import pandas as pd
from datetime import datetime
import yfinance as yf

# Define the stock symbol and start date
symbol = 'AAPL'
start_date = '2001-01-01'

# Set the end date to today's date
end_date = datetime.today().strftime('%Y-%m-%d')

# Download the stock price data using yfinance
df = yf.download(symbol, start=start_date, end=end_date)

df = pd.DataFrame(df)

# convert the date column to datetime
df['Date'] = pd.to_datetime(df.index)

# convert Date to number for day
df['Day'] = (df['Date'] - df['Date'].min()).dt.days

# put date column first
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# remove last coulmn
df = df.iloc[:, :-1]

# display coulmns
print(df.columns)

# Save the data to an Excel file
df.to_csv('stock_prices.csv', index=False)