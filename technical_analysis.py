import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    """Calculates various technical indicators for stock analysis"""
    
    def __init__(self):
        pass
    
    def add_all_indicators(self, df):
        """
        Add all technical indicators to the dataframe
        
        Args:
            df (pandas.DataFrame): OHLCV stock data
        
        Returns:
            pandas.DataFrame: DataFrame with all technical indicators
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying original data
        df_indicators = df.copy()
        
        # Clean the data
        df_indicators = dropna(df_indicators)
        
        try:
            # Trend Indicators
            df_indicators = self._add_trend_indicators(df_indicators)
            
            # Momentum Indicators  
            df_indicators = self._add_momentum_indicators(df_indicators)
            
            # Volatility Indicators
            df_indicators = self._add_volatility_indicators(df_indicators)
            
            # Volume Indicators
            df_indicators = self._add_volume_indicators(df_indicators)
            
            # Others
            df_indicators = self._add_other_indicators(df_indicators)
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            
        return df_indicators
    
    def _add_trend_indicators(self, df):
        """Add trend-based technical indicators"""
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
    
    def _add_momentum_indicators(self, df):
        """Add momentum-based technical indicators"""
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
    
    def _add_volatility_indicators(self, df):
        """Add volatility-based technical indicators"""
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
    
    def _add_volume_indicators(self, df):
        """Add volume-based technical indicators"""
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
    
    def _add_other_indicators(self, df):
        """Add other useful indicators"""
        try:
            # Price change percentage
            df['Price_change'] = df['Close'].pct_change()
            df['Price_change_abs'] = df['Price_change'].abs()
            
            # High-Low percentage
            df['HL_pct'] = (df['High'] - df['Low']) / df['Close']
            
            # Close position within the day's range
            df['Close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Volume change
            df['Volume_change'] = df['Volume'].pct_change()
            
            # Price momentum (5-period)
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            
            # Volatility (rolling standard deviation)
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
        except Exception as e:
            print(f"Error in other indicators: {str(e)}")
            
        return df
    
    def get_signal_summary(self, df):
        """
        Generate a summary of buy/sell signals from technical indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators
        
        Returns:
            dict: Summary of signals
        """
        if df is None or df.empty:
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
            if latest.get('MACD', 0) > latest.get('MACD_signal', 0):
                signals['MACD'] = 'BUY'
            else:
                signals['MACD'] = 'SELL'
            
            # Bollinger Bands signals
            bb_percent = latest.get('BB_percent', 0.5)
            if bb_percent > 0.8:
                signals['BB'] = 'SELL'
            elif bb_percent < 0.2:
                signals['BB'] = 'BUY'
            else:
                signals['BB'] = 'NEUTRAL'
            
            # Moving Average signals
            if latest['Close'] > latest.get('EMA_20', latest['Close']):
                signals['MA'] = 'BUY'
            else:
                signals['MA'] = 'SELL'
            
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
            print(f"Error generating signal summary: {str(e)}")
            
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
        try:
            recent_data = df.tail(window)
            
            support = recent_data['Low'].min()
            resistance = recent_data['High'].max()
            
            # Calculate multiple support/resistance levels
            price_levels = np.concatenate([recent_data['High'], recent_data['Low']])
            price_levels = np.unique(np.round(price_levels, 2))
            price_levels.sort()
            
            return {
                'support': support,
                'resistance': resistance,
                'levels': price_levels
            }
            
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {'support': 0, 'resistance': 0, 'levels': []}
