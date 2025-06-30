import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import streamlit as st

alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY.env")

class StockDataFetcher:
    """Handles fetching stock data from Alpha Vantage API"""
    
    def __init__(self):
        # Get API key from environment variables with fallback
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY.env", "demo")
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_stock_data(self, symbol, interval="1min", outputsize="compact"):
        """
        Fetch intraday stock data from Alpha Vantage
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize (str): 'compact' (latest 100 data points) or 'full'
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Parameters for the API call
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol.upper(),
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            # Make the API request
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                st.error(f"API Error: {data['Error Message']}")
                st.info("This stock symbol may not be supported for intraday data by Alpha Vantage. Try a different symbol (e.g., AAPL, MSFT, TSLA) or check the documentation for supported symbols.")
                return None
            
            if "Note" in data:
                st.warning("API call frequency limit reached. Using demo data.")
                return self._get_demo_data()
            
            # Extract time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                st.warning("No data found. Using demo data for demonstration.")
                return self._get_demo_data()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert data types
            df = df.astype(float)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Ensure we have enough data points
            if len(df) < 50:
                st.warning("Limited data available. Results may be less accurate.")
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return self._get_demo_data()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return self._get_demo_data()
    
    def _get_demo_data(self):
        """
        Generate demo data for testing purposes when API is unavailable
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
