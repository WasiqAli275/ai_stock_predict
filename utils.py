import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def load_or_create_model(model_path, model_class, *args, **kwargs):
    """
    Load an existing model or create a new one
    
    Args:
        model_path (str): Path to the saved model
        model_class (class): Class to instantiate if model doesn't exist
        *args: Arguments for model class
        **kwargs: Keyword arguments for model class
    
    Returns:
        Model instance
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            model = model_class(*args, **kwargs)
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return model_class(*args, **kwargs)

def save_model(model, model_path):
    """
    Save a model to disk
    
    Args:
        model: Model to save
        model_path (str): Path to save the model
    
    Returns:
        bool: Success status
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def format_prediction_time(timeframe):
    """
    Format timeframe for display
    
    Args:
        timeframe (str): Timeframe string
    
    Returns:
        str: Formatted timeframe
    """
    timeframe_map = {
        "1min": "1 minute",
        "5min": "5 minutes", 
        "10min": "10 minutes",
        "15min": "15 minutes",
        "30min": "30 minutes",
        "60min": "1 hour"
    }
    
    return timeframe_map.get(timeframe, timeframe)

def calculate_percentage_change(current, previous):
    """
    Calculate percentage change between two values
    
    Args:
        current (float): Current value
        previous (float): Previous value
    
    Returns:
        float: Percentage change
    """
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def validate_dataframe(df, required_columns=None):
    """
    Validate that a dataframe has required structure
    
    Args:
        df (pandas.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing infinities and NaN values
    
    Args:
        df (pandas.DataFrame): DataFrame to clean
        columns (list): Specific columns to clean (default: all numeric columns)
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    if df is None or df.empty:
        return df
    
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_cleaned.columns:
            # Replace infinities with NaN
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with forward fill, then backward fill
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN, fill with 0
            df_cleaned[col] = df_cleaned[col].fillna(0)
    
    return df_cleaned

def get_market_hours():
    """
    Get current market hours information
    
    Returns:
        dict: Market hours information
    """
    now = datetime.now()
    
    # Simple market hours check (US Eastern Time approximation)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_market_hours = market_open <= now <= market_close and now.weekday() < 5
    
    return {
        "is_open": is_market_hours,
        "market_open": market_open,
        "market_close": market_close,
        "current_time": now
    }

def format_currency(amount, currency="USD"):
    """
    Format amount as currency
    
    Args:
        amount (float): Amount to format
        currency (str): Currency code
    
    Returns:
        str: Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_risk_metrics(returns):
    """
    Calculate basic risk metrics from returns
    
    Args:
        returns (pandas.Series): Series of returns
    
    Returns:
        dict: Risk metrics
    """
    if returns is None or len(returns) == 0:
        return {}
    
    try:
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {}
        
        metrics = {
            "volatility": clean_returns.std(),
            "mean_return": clean_returns.mean(),
            "max_return": clean_returns.max(),
            "min_return": clean_returns.min(),
            "sharpe_ratio": clean_returns.mean() / clean_returns.std() if clean_returns.std() != 0 else 0
        }
        
        # Value at Risk (95% confidence)
        if len(clean_returns) > 0:
            metrics["var_95"] = np.percentile(clean_returns, 5)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        return {}

def create_session_state_defaults():
    """
    Initialize Streamlit session state with default values
    """
    defaults = {
        "selected_stock": "AAPL",
        "prediction_history": [],
        "model_trained": False,
        "last_prediction_time": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def log_prediction(symbol, prediction_result):
    """
    Log prediction to session state history
    
    Args:
        symbol (str): Stock symbol
        prediction_result (dict): Prediction results
    """
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    
    log_entry = {
        "timestamp": datetime.now(),
        "symbol": symbol,
        "prediction": prediction_result.get("prediction"),
        "confidence": prediction_result.get("confidence"),
        "action": prediction_result.get("action")
    }
    
    st.session_state.prediction_history.append(log_entry)
    
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]

def get_prediction_accuracy():
    """
    Calculate prediction accuracy from session history (simplified)
    
    Returns:
        dict: Accuracy statistics
    """
    if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
        return {"total_predictions": 0, "accuracy": 0}
    
    history = st.session_state.prediction_history
    
    # This is a simplified calculation
    # In a real system, you'd need to track actual outcomes
    total = len(history)
    
    # Mock accuracy based on confidence levels (for demonstration)
    high_confidence_predictions = sum(1 for p in history if p.get("confidence", 0) > 0.7)
    estimated_accuracy = min(0.9, (high_confidence_predictions / total) * 0.8 + 0.5) if total > 0 else 0
    
    return {
        "total_predictions": total,
        "accuracy": estimated_accuracy,
        "high_confidence_count": high_confidence_predictions
    }

# Utility functions for data validation and error handling
def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def safe_percentage(value, total, default=0):
    """Safely calculate percentage"""
    return safe_divide(value * 100, total, default)

def truncate_text(text, max_length=100):
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
