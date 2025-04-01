import os  # Ensure os is imported
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Define ETF list
ETF_LIST = ["VUSA.DE", "VWCE.DE", "IS3N.DE", "EXS1.DE", "VECP.DE"]
CACHE_DIR = "cache"

def fetch_single_etf(ticker, start_date=None, end_date=None):
    """Fetch data for a single ETF or index"""
    try:
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        if not data.empty:
            return data['Close']
        print(f"No data found for {ticker}")
        return None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def fetch_etf_data(start_date="2020-01-01", end_date=None):
    """Fetch ETF data for all tickers in ETF_LIST"""
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    all_data = pd.DataFrame()
    
    for etf in ETF_LIST:
        series = fetch_single_etf(etf, start_date, end_date)
        if series is not None:
            all_data[etf] = series
            print(f"Successfully fetched data for {etf}")
    
    return all_data

def get_dax_data(start_date="2020-01-01", end_date=None):
    """Fetch DAX index data"""
    return fetch_single_etf("^GDAXI", start_date, end_date)

def load_etf_data(filename="etf_prices.csv", refresh=False):
    """Load ETF data from CSV or fetch new data"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    filepath = os.path.join(CACHE_DIR, filename)
    
    try:
        if not refresh and os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if all(etf in df.columns for etf in ETF_LIST):
                return df
    except Exception as e:
        print(f"Error loading from CSV: {e}")
    
    df = fetch_etf_data()
    if not df.empty:
        df.to_csv(filepath)
        print(f"Saved new data to {filepath}")
    return df

if __name__ == "__main__":
    df = load_etf_data(refresh=True)
    if not df.empty:
        print("\nFirst 5 rows of data:")
        print(df.head())
    else:
        print("Failed to load ETF data")