import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import streamlit as st # Ensure Streamlit is imported for the decorator
import warnings
warnings.filterwarnings('ignore')

# --- Top-level helper functions for adding indicators ---

def _add_trend_indicators_static(df_input):
    """Add trend-based technical indicators."""
    df = df_input # Work directly on the df passed (copy is made in core function)
    try:
        # Simple Moving Averages
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Exponential Moving Averages
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_histogram'] = ta.trend.macd(df['Close'])
        
        # Average Directional Index
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.psar_down(df['High'], df['Low'], df['Close'])
        
    except Exception as e:
        print(f"Error in trend indicators: {str(e)}")
    return df

def _add_momentum_indicators_static(df_input):
    """Add momentum-based technical indicators."""
    df = df_input
    try:
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # Stochastic Oscillator
        df['STOCH_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['STOCH_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

        # Williams %R
        df['WILLIAMS_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

        # Rate of Change
        df['ROC'] = ta.momentum.roc(df['Close'])

        # Money Flow Index
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

    except Exception as e:
        print(f"Error in momentum indicators: {str(e)}")
    return df

def _add_volatility_indicators_static(df_input):
    """Add volatility-based technical indicators"""
    df = df_input
    try:
        # Bollinger Bands
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_middle'] = ta.volatility.bollinger_mavg(df['Close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_width'] = ta.volatility.bollinger_wband(df['Close'])
        df['BB_percent'] = ta.volatility.bollinger_pband(df['Close'])

        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Keltner Channels
        df['KC_upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
        df['KC_middle'] = ta.volatility.keltner_channel_mband(df['High'], df['Low'], df['Close'])
        df['KC_lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])

    except Exception as e:
        print(f"Error in volatility indicators: {str(e)}")
    return df

def _add_volume_indicators_static(df_input):
    """Add volume-based technical indicators"""
    df = df_input
    try:
        # On-Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

        # Accumulation/Distribution Line
        df['ADL'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Chaikin Money Flow
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volume SMA
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)

    except Exception as e:
        print(f"Error in volume indicators: {str(e)}")
    return df

def _add_other_indicators_static(df_input):
    """Add other useful indicators"""
    df = df_input
    try:
        # Price change percentage
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_abs'] = df['Price_change'].abs() # Keep this if used, else remove

        # High-Low percentage
        df['HL_pct'] = (df['High'] - df['Low']) / df['Close']

        # Close position within the day's range (handle potential division by zero)
        range_hl = df['High'] - df['Low']
        df['Close_position'] = ((df['Close'] - df['Low']) / range_hl).replace([np.inf, -np.inf], np.nan)

        # Volume change
        df['Volume_change'] = df['Volume'].pct_change()

        # Price momentum (5-period)
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1

        # Volatility (rolling standard deviation)
        df['Volatility'] = df['Close'].rolling(window=20).std()

    except Exception as e:
        print(f"Error in other indicators: {str(e)}")
    return df

# --- Core cached function for calculating all indicators ---
# @st.cache_data # Temporarily disabled for debugging
def _calculate_indicators_core(input_df):
    """
    Core logic for adding all technical indicators, cached.
    Operates on a copy of the input DataFrame.
    """
    if input_df is None or input_df.empty:
        return input_df
    
    # Make a copy to avoid modifying original cached data if input_df is already from cache
    df_indicators = input_df.copy()
    
    # Clean the data (dropna might be aggressive, consider ffill/bfill first if needed)
    # For now, keeping original dropna as per existing logic.
    df_indicators = dropna(df_indicators)
    
    if df_indicators.empty: # If dropna results in empty df
        # st.warning("DataFrame became empty after dropna. Check input data for excessive NaNs.")
        return df_indicators # Or return input_df if preferred

    try:
        # Call static helper functions
        df_indicators = _add_trend_indicators_static(df_indicators)
        df_indicators = _add_momentum_indicators_static(df_indicators)
        df_indicators = _add_volatility_indicators_static(df_indicators)
        df_indicators = _add_volume_indicators_static(df_indicators)
        df_indicators = _add_other_indicators_static(df_indicators)

    except Exception as e:
        # This top-level catch might be redundant if individual helpers handle their errors,
        # but can be a fallback.
        st.error(f"Error calculating indicators: {str(e)}") # Use st.error for visibility in app
        # Decide on return: return partially processed df_indicators or original input_df or None
        return input_df # Fallback to original df to avoid breaking downstream if partial is bad

    return df_indicators

class TechnicalAnalyzer:
    """Calculates various technical indicators for stock analysis"""
    
    def __init__(self):
        # No state needed for this class anymore if all logic is in static/top-level functions
        pass
    
    def add_all_indicators(self, df):
        """
        Add all technical indicators to the dataframe.
        This method now calls the cached top-level function.
        """
        return _calculate_indicators_core(df)
    
    # The individual _add_*_indicators methods are now top-level static-like functions.
    # get_signal_summary and calculate_support_resistance can remain methods
    # or also be refactored if they don't rely on instance state.
    # For now, leaving them as methods.

    def get_signal_summary(self, df):
        """
        Generate a summary of buy/sell signals from technical indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators
        
        Returns:
            dict: Summary of signals
        """
        if df is None or df.empty or df.iloc[-1].isnull().all(): # Check if last row is all NaN
            # st.warning("Cannot generate signal summary: DataFrame is empty or latest data is invalid.")
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        try:
            # RSI signals
            if latest.get('RSI', 50) > 70:
                signals['RSI'] = 'SELL'
            elif latest.get('RSI', 50) < 30:
                signals['RSI'] = 'BUY'
            else:
                signals['RSI'] = 'NEUTRAL'
            
            # MACD signals
            if latest.get('MACD', 0) > latest.get('MACD_signal', 0): # Ensure MACD_signal might exist
                signals['MACD'] = 'BUY'
            elif latest.get('MACD', 0) < latest.get('MACD_signal', 0):
                signals['MACD'] = 'SELL'
            else: # Handles cases where one might be NaN or they are equal
                signals['MACD'] = 'NEUTRAL'

            # Bollinger Bands signals
            bb_percent = latest.get('BB_percent', 0.5)
            if bb_percent > 0.8: # Using 0.8 and 0.2 as common thresholds for overbought/oversold
                signals['BB'] = 'SELL'
            elif bb_percent < 0.2:
                signals['BB'] = 'BUY'
            else:
                signals['BB'] = 'NEUTRAL'
            
            # Moving Average signals (using EMA_20 as an example, ensure it exists)
            # Original code used EMA_20, but it's not calculated. Let's use EMA_12 or EMA_26.
            # Using EMA_26 as it's a common component of MACD.
            if 'EMA_26' in latest and not pd.isna(latest['EMA_26']):
                if latest['Close'] > latest['EMA_26']:
                    signals['MA'] = 'BUY'
                else:
                    signals['MA'] = 'SELL'
            else:
                signals['MA'] = 'NEUTRAL' # Default if indicator is missing
            
            # Overall signal
            buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
            sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
            
            if buy_signals > sell_signals:
                signals['OVERALL'] = 'BUY'
            elif sell_signals > buy_signals:
                signals['OVERALL'] = 'SELL'
            else:
                signals['OVERALL'] = 'NEUTRAL'
                
        except Exception as e:
            print(f"Error generating signal summary: {str(e)}") # Keep print for server logs
            # st.warning(f"Could not generate signal summary: {str(e)}") # Optional UI warning
            
        return signals
    
    def calculate_support_resistance(self, df, window=20):
        """
        Calculate support and resistance levels
        
        Args:
            df (pandas.DataFrame): OHLCV data
            window (int): Lookback window for calculation
        
        Returns:
            dict: Support and resistance levels
        """
        if df is None or df.empty or len(df) < window : # Ensure enough data for window
            # st.warning("Not enough data to calculate support/resistance.")
            return {'support': np.nan, 'resistance': np.nan, 'levels': []}

        try:
            recent_data = df.tail(window)
            
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            
            # Calculate multiple support/resistance levels
            # Ensure data is not all NaN before trying to work with it
            if recent_data[['High', 'Low']].isnull().all().all():
                return {'support': np.nan, 'resistance': np.nan, 'levels': []}

            price_levels = np.concatenate([recent_data['High'].dropna(), recent_data['Low'].dropna()])
            if len(price_levels) == 0:
                 return {'support': support if not pd.isna(support) else np.nan,
                         'resistance': resistance if not pd.isna(resistance) else np.nan,
                         'levels': []}

            price_levels = np.unique(np.round(price_levels, 2))
            price_levels.sort()
            
            return {
                'support': support,
                'resistance': resistance,
                'levels': price_levels.tolist() # Convert to list for JSON compatibility if needed
            }
            
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {'support': np.nan, 'resistance': np.nan, 'levels': []}
