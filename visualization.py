import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartVisualizer:
    """Creates interactive charts for stock analysis"""
    
    def __init__(self):
        pass
    
    def create_candlestick_chart(self, df, symbol, chart_type="Candlestick"):
        """
        Create an interactive candlestick chart with technical indicators
        matching professional trading interface style
        
        Args:
            df (pandas.DataFrame): OHLCV data with indicators
            symbol (str): Stock symbol
            chart_type (str): Type of chart ("Candlestick", "Line", "OHLC")
        
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        if df is None or df.empty:
            return self._create_empty_chart()
        
        try:
            # If candlestick fails, automatically fallback to line chart
            if chart_type == "Candlestick":
                try:
                    return self._create_candlestick_chart(df, symbol)
                except Exception as e:
                    print(f"Candlestick chart failed, falling back to line chart: {e}")
                    return self._create_line_chart(df, symbol)
            elif chart_type == "Line":
                return self._create_line_chart(df, symbol)
            else:
                return self._create_line_chart(df, symbol)
        except Exception as e:
            print(f"Chart creation failed: {e}")
            return self._create_simple_line_chart(df, symbol)
    
    def _create_candlestick_chart(self, df, symbol):
        """Create candlestick chart with full features"""
        try:
            # Create subplots with professional layout
            price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100) if len(df) > 1 else 0
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=[
                    f'{symbol} - ${df["Close"].iloc[-1]:.2f} ({price_change:+.2f}%)',
                    'Volume',
                    'RSI (14)',
                    'MACD (12,26,9)'
                ],
                row_heights=[0.6, 0.15, 0.125, 0.125]
            )
            
            # Main candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350',
                    increasing_fillcolor='#26a69a',
                    decreasing_fillcolor='#ef5350',
                    line=dict(width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'EMA_12' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_12'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color='orange', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            if 'EMA_26' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_26'],
                        mode='lines',
                        name='EMA 26',
                        line=dict(color='purple', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=0.5
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        opacity=0.5
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_middle'],
                        mode='lines',
                        name='BB Middle',
                        line=dict(color='gray', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # RSI chart
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#ffc107', width=2)
                    ),
                    row=3, col=1
                )
                
                # RSI reference lines (using shapes instead of add_hline for subplot compatibility)
                fig.add_shape(
                    type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                    line=dict(color="red", width=1, dash="dash"), opacity=0.5,
                    row=3, col=1
                )
                fig.add_shape(
                    type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                    line=dict(color="green", width=1, dash="dash"), opacity=0.5,
                    row=3, col=1
                )
                fig.add_shape(
                    type="line", x0=df.index[0], x1=df.index[-1], y0=50, y1=50,
                    line=dict(color="gray", width=1, dash="dot"), opacity=0.3,
                    row=3, col=1
                )
            
            # MACD chart
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD_signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=4, col=1
                )
                
                # MACD histogram
                if 'MACD_histogram' in df.columns:
                    colors_macd = ['green' if val >= 0 else 'red' for val in df['MACD_histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['MACD_histogram'],
                            name='MACD Histogram',
                            marker_color=colors_macd,
                            opacity=0.6
                        ),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Technical Analysis Dashboard",
                xaxis_title="Time",
                template="plotly_dark",
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            # Remove x-axis labels for all but bottom chart
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_xaxes(showticklabels=False, row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=3, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating candlestick chart: {str(e)}")
            return self._create_empty_chart()
    
    def _create_line_chart(self, df, symbol):
        """Create a simple line chart with technical indicators"""
        try:
            price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100) if len(df) > 1 else 0
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=[
                    f'{symbol} - ${df["Close"].iloc[-1]:.2f} ({price_change:+.2f}%)',
                    'Volume',
                    'RSI (14)',
                    'MACD (12,26,9)'
                ],
                row_heights=[0.6, 0.15, 0.125, 0.125]
            )
            
            # Main price line chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00ff88', width=2)
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'EMA_12' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_12'],
                        mode='lines',
                        name='EMA 12',
                        line=dict(color='orange', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            if 'EMA_26' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_26'],
                        mode='lines',
                        name='EMA 26',
                        line=dict(color='purple', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Volume bars
            if 'Volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume',
                        marker_color='rgba(128, 128, 128, 0.5)'
                    ),
                    row=2, col=1
                )
            
            # RSI chart
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=1
                )
            
            # MACD chart
            if 'MACD' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=4, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Line Chart Analysis",
                template="plotly_dark",
                height=800,
                showlegend=True
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating line chart: {str(e)}")
            return self._create_simple_line_chart(df, symbol)
    
    def _create_simple_line_chart(self, df, symbol):
        """Create a basic line chart as final fallback"""
        try:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00ff88', width=2)
                )
            )
            
            fig.update_layout(
                title=f"{symbol} - Stock Price",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating simple chart: {str(e)}")
            return self._create_empty_chart()
    
    def create_prediction_chart(self, df, prediction_result):
        """
        Create a chart showing prediction visualization
        
        Args:
            df (pandas.DataFrame): Historical data
            prediction_result (dict): Prediction results
        
        Returns:
            plotly.graph_objects.Figure: Prediction chart
        """
        try:
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='white', width=2)
                )
            )
            
            # Prediction point
            last_price = df['Close'].iloc[-1]
            last_time = df.index[-1]
            
            prediction = prediction_result.get('prediction', 'SUSTAIN')
            confidence = prediction_result.get('confidence', 0.5)
            
            # Show prediction direction
            if prediction == 'UP':
                future_price = last_price * 1.01  # 1% increase
                color = 'green'
                symbol = 'triangle-up'
            elif prediction == 'DOWN':
                future_price = last_price * 0.99  # 1% decrease
                color = 'red'
                symbol = 'triangle-down'
            else:
                future_price = last_price
                color = 'yellow'
                symbol = 'circle'
            
            # Add prediction point
            fig.add_trace(
                go.Scatter(
                    x=[last_time],
                    y=[future_price],
                    mode='markers',
                    name=f'Prediction: {prediction}',
                    marker=dict(
                        color=color,
                        size=15,
                        symbol=symbol,
                        line=dict(color='white', width=2)
                    )
                )
            )
            
            # Add confidence interval
            confidence_range = last_price * 0.005 * confidence  # Scale with confidence
            
            fig.add_trace(
                go.Scatter(
                    x=[last_time, last_time],
                    y=[future_price - confidence_range, future_price + confidence_range],
                    mode='lines',
                    name=f'Confidence: {confidence:.1%}',
                    line=dict(color=color, width=4, dash='dash'),
                    opacity=0.6
                )
            )
            
            fig.update_layout(
                title="Price Prediction Visualization",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                template="plotly_dark",
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating prediction chart: {str(e)}")
            return self._create_empty_chart()
    
    def create_technical_indicators_chart(self, df):
        """
        Create a chart focused on technical indicators
        
        Args:
            df (pandas.DataFrame): Data with technical indicators
        
        Returns:
            plotly.graph_objects.Figure: Technical indicators chart
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['RSI & Stochastic', 'MACD', 'Bollinger Bands %B', 'Volume Indicators'],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # RSI and Stochastic
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')),
                    row=1, col=1
                )
            
            if 'STOCH_k' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['STOCH_k'], name='Stochastic %K', line=dict(color='blue')),
                    row=1, col=1
                )
            
            # MACD
            if 'MACD' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
                    row=1, col=2
                )
            
            if 'MACD_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')),
                    row=1, col=2
                )
            
            # Bollinger Bands %B
            if 'BB_percent' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_percent'], name='BB %B', line=dict(color='purple')),
                    row=2, col=1
                )
            
            # Volume indicators
            if 'Volume' in df.columns:
                fig.add_trace(
                    go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.6),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Technical Indicators Dashboard",
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating technical indicators chart: {str(e)}")
            return self._create_empty_chart()
    
    def _create_empty_chart(self):
        """Create an empty chart as fallback"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
