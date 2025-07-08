import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import streamlit as st

alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "1JUV860DC7YJRVYM")

# Helper function to generate demo data, used by the cached core function
def _generate_demo_stock_data():
    """
    Generate demo data for testing purposes.
    Moved out of class to be callable by the top-level cached function.
    """
    np.random.seed(42)
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=100)
    date_range = pd.date_range(start=start_time, end=end_time, freq='1T')
    base_price = 150.0
    returns = np.random.normal(0, 0.002, len(date_range))
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(1000, 10000)
        data.append({
            'Open': open_price,
            'High': max(open_price, high, close_price),
            'Low': min(open_price, low, close_price),
            'Close': close_price,
            'Volume': volume
        })
    df = pd.DataFrame(data, index=date_range)
    return df

@st.cache_data
def _fetch_stock_data_core(symbol, interval, outputsize, api_key_to_use, base_url_to_use):
    """
    Core logic for fetching intraday stock data from Alpha Vantage, cached.
    This function is now top-level to ensure caching is based purely on its arguments.
    """
    try:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol.upper(),
            'interval': interval,
            'apikey': api_key_to_use,
            'outputsize': outputsize,
            'datatype': 'json'
        }
        response = requests.get(base_url_to_use, params=params, timeout=30)
        data = response.json()

        if "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        if "Note" in data: # Typically indicates API limit reached
            st.warning("API call frequency limit reached or other API note. Using demo data.")
            return _generate_demo_stock_data()

        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            st.warning(f"No data for '{time_series_key}' found for symbol {symbol}. Using demo data.")
            return _generate_demo_stock_data()

        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if len(df) < 50: # Check after successful fetch and parsing
            st.warning("Limited data available post-fetch. Results may be less accurate.")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}. Using demo data.")
        return _generate_demo_stock_data()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}. Using demo data.")
        return _generate_demo_stock_data()

class StockDataFetcher:
    """Handles fetching stock data from Alpha Vantage API"""
    
    def __init__(self):
        # Get API key from environment variables with fallback
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo") # Fallback to "demo" if not set
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_stock_data(self, symbol, interval="1min", outputsize="compact"):
        """
        Fetch intraday stock data from Alpha Vantage.
        This method now calls a cached top-level function.
        """
        # The API key from self.api_key (which could be 'demo') is passed explicitly
        return _fetch_stock_data_core(symbol, interval, outputsize, self.api_key, self.base_url)
    
    def _get_demo_data(self):
        """
        Generate demo data for testing purposes when API is unavailable.
        This method is now a wrapper around the global helper.
        """
        # Generate realistic stock data
        np.random.seed(42)  # For reproducible results
        
        # Create datetime index for the last 100 minutes
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=100)
        date_range = pd.date_range(start=start_time, end=end_time, freq='1T')
        
        # Generate realistic price movements
        base_price = 150.0
        returns = np.random.normal(0, 0.002, len(date_range))  # 0.2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC from the base price
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'Open': open_price,
                'High': max(open_price, high, close_price),
                'Low': min(open_price, low, close_price),
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=date_range)
        return df
    
    def get_daily_data(self, symbol, outputsize="compact"):
        """
        Fetch daily stock data for longer-term analysis
        
        Args:
            symbol (str): Stock symbol
            outputsize (str): 'compact' or 'full'
        
        Returns:
            pandas.DataFrame: Daily stock data
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol.upper(),
                'apikey': self.api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                return None
            
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching daily data: {str(e)}")
            return None
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Try to fetch a small amount of data
            data = self.get_stock_data(symbol, outputsize="compact")
            return data is not None and not data.empty
        except:
            return False
