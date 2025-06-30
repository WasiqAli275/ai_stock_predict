import pandas as pd
import numpy as np
from typing import List, Dict

class PatternDetector:
    """Detects candlestick patterns and chart patterns"""
    
    def __init__(self):
        self.patterns = []
    
    def detect_patterns(self, df):
        """
        Detect various candlestick and chart patterns
        
        Args:
            df (pandas.DataFrame): OHLCV data with technical indicators
        
        Returns:
            list: List of detected patterns
        """
        if df is None or df.empty or len(df) < 10:
            return []
        
        patterns = []
        
        try:
            # Candlestick patterns
            patterns.extend(self._detect_candlestick_patterns(df))
            
            # Chart patterns
            patterns.extend(self._detect_chart_patterns(df))
            
            # Technical patterns
            patterns.extend(self._detect_technical_patterns(df))
            
        except Exception as e:
            print(f"Error detecting patterns: {str(e)}")
        
        return patterns
    
    def _detect_candlestick_patterns(self, df):
        """Detect single and multi-candle patterns"""
        patterns = []
        
        try:
            # Get recent candles (last 10)
            recent = df.tail(10).copy()
            
            if len(recent) < 3:
                return patterns
            
            # Calculate candle properties
            recent['body'] = abs(recent['Close'] - recent['Open'])
            recent['upper_shadow'] = recent['High'] - recent[['Open', 'Close']].max(axis=1)
            recent['lower_shadow'] = recent[['Open', 'Close']].min(axis=1) - recent['Low']
            recent['total_range'] = recent['High'] - recent['Low']
            recent['is_bullish'] = recent['Close'] > recent['Open']
            recent['is_bearish'] = recent['Close'] < recent['Open']
            
            # Single candle patterns
            for i in range(len(recent)):
                candle = recent.iloc[i]
                
                # Doji
                if self._is_doji(candle):
                    patterns.append(f"Doji detected at {candle.name.strftime('%H:%M')}")
                
                # Hammer
                if self._is_hammer(candle):
                    patterns.append(f"Hammer pattern at {candle.name.strftime('%H:%M')}")
                
                # Shooting Star
                if self._is_shooting_star(candle):
                    patterns.append(f"Shooting Star at {candle.name.strftime('%H:%M')}")
                
                # Spinning Top
                if self._is_spinning_top(candle):
                    patterns.append(f"Spinning Top at {candle.name.strftime('%H:%M')}")
            
            # Multi-candle patterns
            if len(recent) >= 2:
                patterns.extend(self._detect_two_candle_patterns(recent))
            
            if len(recent) >= 3:
                patterns.extend(self._detect_three_candle_patterns(recent))
                
        except Exception as e:
            print(f"Error in candlestick pattern detection: {str(e)}")
        
        return patterns
    
    def _is_doji(self, candle):
        """Check if candle is a Doji"""
        body_size = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return False
        
        return body_size / total_range < 0.1
    
    def _is_hammer(self, candle):
        """Check if candle is a Hammer"""
        body = abs(candle['Close'] - candle['Open'])
        lower_shadow = candle['lower_shadow']
        upper_shadow = candle['upper_shadow']
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return False
        
        # Hammer conditions
        return (lower_shadow > 2 * body and 
                upper_shadow < body * 0.5 and
                lower_shadow > 0.6 * total_range)
    
    def _is_shooting_star(self, candle):
        """Check if candle is a Shooting Star"""
        body = abs(candle['Close'] - candle['Open'])
        lower_shadow = candle['lower_shadow']
        upper_shadow = candle['upper_shadow']
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return False
        
        # Shooting star conditions
        return (upper_shadow > 2 * body and 
                lower_shadow < body * 0.5 and
                upper_shadow > 0.6 * total_range)
    
    def _is_spinning_top(self, candle):
        """Check if candle is a Spinning Top"""
        body = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        upper_shadow = candle['upper_shadow']
        lower_shadow = candle['lower_shadow']
        
        if total_range == 0:
            return False
        
        # Spinning top conditions
        return (body < 0.3 * total_range and
                upper_shadow > 0.1 * total_range and
                lower_shadow > 0.1 * total_range)
    
    def _detect_two_candle_patterns(self, df):
        """Detect two-candle patterns"""
        patterns = []
        
        try:
            for i in range(1, len(df)):
                prev_candle = df.iloc[i-1]
                curr_candle = df.iloc[i]
                
                # Bullish Engulfing
                if self._is_bullish_engulfing(prev_candle, curr_candle):
                    patterns.append(f"Bullish Engulfing pattern at {curr_candle.name.strftime('%H:%M')}")
                
                # Bearish Engulfing
                if self._is_bearish_engulfing(prev_candle, curr_candle):
                    patterns.append(f"Bearish Engulfing pattern at {curr_candle.name.strftime('%H:%M')}")
                
                # Piercing Pattern
                if self._is_piercing_pattern(prev_candle, curr_candle):
                    patterns.append(f"Piercing Pattern at {curr_candle.name.strftime('%H:%M')}")
                
                # Dark Cloud Cover
                if self._is_dark_cloud_cover(prev_candle, curr_candle):
                    patterns.append(f"Dark Cloud Cover at {curr_candle.name.strftime('%H:%M')}")
                
        except Exception as e:
            print(f"Error in two-candle pattern detection: {str(e)}")
        
        return patterns
    
    def _is_bullish_engulfing(self, prev, curr):
        """Check for Bullish Engulfing pattern"""
        return (prev['is_bearish'] and 
                curr['is_bullish'] and
                curr['Open'] < prev['Close'] and
                curr['Close'] > prev['Open'])
    
    def _is_bearish_engulfing(self, prev, curr):
        """Check for Bearish Engulfing pattern"""
        return (prev['is_bullish'] and 
                curr['is_bearish'] and
                curr['Open'] > prev['Close'] and
                curr['Close'] < prev['Open'])
    
    def _is_piercing_pattern(self, prev, curr):
        """Check for Piercing Pattern"""
        return (prev['is_bearish'] and 
                curr['is_bullish'] and
                curr['Open'] < prev['Low'] and
                curr['Close'] > (prev['Open'] + prev['Close']) / 2)
    
    def _is_dark_cloud_cover(self, prev, curr):
        """Check for Dark Cloud Cover"""
        return (prev['is_bullish'] and 
                curr['is_bearish'] and
                curr['Open'] > prev['High'] and
                curr['Close'] < (prev['Open'] + prev['Close']) / 2)
    
    def _detect_three_candle_patterns(self, df):
        """Detect three-candle patterns"""
        patterns = []
        
        try:
            for i in range(2, len(df)):
                candle1 = df.iloc[i-2]
                candle2 = df.iloc[i-1]
                candle3 = df.iloc[i]
                
                # Morning Star
                if self._is_morning_star(candle1, candle2, candle3):
                    patterns.append(f"Morning Star pattern at {candle3.name.strftime('%H:%M')}")
                
                # Evening Star
                if self._is_evening_star(candle1, candle2, candle3):
                    patterns.append(f"Evening Star pattern at {candle3.name.strftime('%H:%M')}")
                
                # Three White Soldiers
                if self._is_three_white_soldiers(candle1, candle2, candle3):
                    patterns.append(f"Three White Soldiers at {candle3.name.strftime('%H:%M')}")
                
                # Three Black Crows
                if self._is_three_black_crows(candle1, candle2, candle3):
                    patterns.append(f"Three Black Crows at {candle3.name.strftime('%H:%M')}")
                
        except Exception as e:
            print(f"Error in three-candle pattern detection: {str(e)}")
        
        return patterns
    
    def _is_morning_star(self, c1, c2, c3):
        """Check for Morning Star pattern"""
        return (c1['is_bearish'] and
                c3['is_bullish'] and
                c2['body'] < min(c1['body'], c3['body']) * 0.5 and
                c3['Close'] > (c1['Open'] + c1['Close']) / 2)
    
    def _is_evening_star(self, c1, c2, c3):
        """Check for Evening Star pattern"""
        return (c1['is_bullish'] and
                c3['is_bearish'] and
                c2['body'] < min(c1['body'], c3['body']) * 0.5 and
                c3['Close'] < (c1['Open'] + c1['Close']) / 2)
    
    def _is_three_white_soldiers(self, c1, c2, c3):
        """Check for Three White Soldiers pattern"""
        return (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'] and
                c2['Close'] > c1['Close'] and c3['Close'] > c2['Close'] and
                c2['Open'] > c1['Open'] and c3['Open'] > c2['Open'])
    
    def _is_three_black_crows(self, c1, c2, c3):
        """Check for Three Black Crows pattern"""
        return (c1['is_bearish'] and c2['is_bearish'] and c3['is_bearish'] and
                c2['Close'] < c1['Close'] and c3['Close'] < c2['Close'] and
                c2['Open'] < c1['Open'] and c3['Open'] < c2['Open'])
    
    def _detect_chart_patterns(self, df):
        """Detect chart patterns like support/resistance breaks"""
        patterns = []
        
        try:
            if len(df) < 20:
                return patterns
            
            recent = df.tail(20)
            
            # Support and resistance levels
            support = recent['Low'].min()
            resistance = recent['High'].max()
            current_price = df['Close'].iloc[-1]
            
            # Support break
            if current_price < support * 1.001:  # Within 0.1%
                patterns.append("Price testing support level")
            
            # Resistance break
            if current_price > resistance * 0.999:  # Within 0.1%
                patterns.append("Price testing resistance level")
            
            # Breakout patterns
            recent_range = resistance - support
            if recent_range < current_price * 0.02:  # Range less than 2%
                patterns.append("Tight consolidation - potential breakout")
                
        except Exception as e:
            print(f"Error in chart pattern detection: {str(e)}")
        
        return patterns
    
    def _detect_technical_patterns(self, df):
        """Detect technical indicator patterns"""
        patterns = []
        
        try:
            if len(df) < 5:
                return patterns
            
            recent = df.tail(5)
            
            # RSI patterns
            if 'RSI' in df.columns:
                rsi_values = recent['RSI'].dropna()
                if len(rsi_values) >= 2:
                    if rsi_values.iloc[-1] > 70:
                        patterns.append("RSI indicates overbought condition")
                    elif rsi_values.iloc[-1] < 30:
                        patterns.append("RSI indicates oversold condition")
                    
                    # RSI divergence (simplified)
                    if len(rsi_values) >= 3:
                        if (rsi_values.iloc[-1] > rsi_values.iloc[-2] > rsi_values.iloc[-3] and
                            recent['Close'].iloc[-1] < recent['Close'].iloc[-3]):
                            patterns.append("Bullish RSI divergence detected")
            
            # MACD patterns
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd_recent = recent[['MACD', 'MACD_signal']].dropna()
                if len(macd_recent) >= 2:
                    if (macd_recent['MACD'].iloc[-2] <= macd_recent['MACD_signal'].iloc[-2] and
                        macd_recent['MACD'].iloc[-1] > macd_recent['MACD_signal'].iloc[-1]):
                        patterns.append("MACD bullish crossover")
                    elif (macd_recent['MACD'].iloc[-2] >= macd_recent['MACD_signal'].iloc[-2] and
                          macd_recent['MACD'].iloc[-1] < macd_recent['MACD_signal'].iloc[-1]):
                        patterns.append("MACD bearish crossover")
            
            # Bollinger Bands patterns
            if 'BB_percent' in df.columns:
                bb_recent = recent['BB_percent'].dropna()
                if len(bb_recent) >= 1:
                    if bb_recent.iloc[-1] > 0.95:
                        patterns.append("Price at upper Bollinger Band")
                    elif bb_recent.iloc[-1] < 0.05:
                        patterns.append("Price at lower Bollinger Band")
                        
        except Exception as e:
            print(f"Error in technical pattern detection: {str(e)}")
        
        return patterns
