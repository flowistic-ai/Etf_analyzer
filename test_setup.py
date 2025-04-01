import dash
import pandas as pd
import yfinance as yf

# Print library versions to confirm installation
print("Dash version:", dash.__version__)
print("Pandas version:", pd.__version__)

# Test data fetching with a real European ETF (iShares Core DAX UCITS ETF)
data = yf.download("EXS1.DE", period="1d")

# Print the first few rows of data
print("Sample ETF data:\n", data.head())