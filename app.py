import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import time
import zipfile
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_fetcher import StockDataFetcher
from technical_analysis import TechnicalAnalyzer
from ml_predictor import MLPredictor
from pattern_detector import PatternDetector
from visualization import ChartVisualizer
from explanation_engine import ExplanationEngine
from utils import load_or_create_model, format_prediction_time
from database import get_database_manager, init_database

def create_source_code_zip():
    """Create a zip file with all source code files"""
    zip_buffer = io.BytesIO()
    
    # List of source files to include
    source_files = [
        'app.py',
        'data_fetcher.py',
        'technical_analysis.py',
        'ml_predictor.py',
        'pattern_detector.py',
        'visualization.py',
        'explanation_engine.py',
        'database.py',
        'utils.py',
        'requirements.md',
        'replit.md',
        '.streamlit/config.toml'
    ]
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_files:
            if os.path.exists(file_path):
                zip_file.write(file_path)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #00ff88, #0066cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-weight: bold;
    font-size: 1.2rem;
}
.buy-signal {
    background-color: rgba(0, 255, 136, 0.2);
    border: 2px solid #00ff88;
}
.sell-signal {
    background-color: rgba(255, 67, 54, 0.2);
    border: 2px solid #ff4336;
}
.hold-signal {
    background-color: rgba(255, 193, 7, 0.2);
    border: 2px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize database connection
    if 'db_initialized' not in st.session_state:
        try:
            init_database()
            st.session_state.db_initialized = True
            st.success("Database connected successfully!", icon="🗄️")
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            st.session_state.db_initialized = False
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)")
        
        # Prediction timeframe
        timeframe = st.selectbox(
            "Prediction Timeframe",
            ["1min", "5min", "10min"],
            help="Select how far ahead to predict"
        )
        
        # Model selection
        model_type = st.selectbox(
            "ML Model",
            ["Random Forest", "XGBoost", "SVM"],
            help="Choose the machine learning model"
        )
        
        # Educational mode
        education_mode = st.checkbox("🎓 Educational Mode", value=True, help="Show explanations for beginners")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)
            use_sentiment = st.checkbox("Include Sentiment Analysis", value=False)
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        
        # Predict button
        predict_button = st.button("🚀 Predict Stock Movement", type="primary")
        
        # Database Statistics
        if st.session_state.get('db_initialized', False):
            with st.expander("📊 Database Statistics"):
                try:
                    db = get_database_manager()
                    stats = db.get_database_stats()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Stock Records", stats.get('stock_data_count', 0))
                        st.metric("Predictions", stats.get('predictions_count', 0))
                        st.metric("Patterns", stats.get('patterns_count', 0))
                    
                    with col2:
                        st.metric("Indicators", stats.get('indicators_count', 0))
                        st.metric("Sessions", stats.get('sessions_count', 0))
                        unique_symbols = stats.get('unique_symbols', [])
                        st.metric("Tracked Symbols", len(unique_symbols))
                    
                    if unique_symbols:
                        st.write("**Symbols:** " + ", ".join(unique_symbols[:10]))
                        
                except Exception as e:
                    st.error(f"Database stats error: {str(e)}")
        
        # Source Code Download Section
        st.sidebar.markdown("---")
        st.sidebar.subheader("💻 Source Code")
        
        # Create download button for source code
        try:
            zip_data = create_source_code_zip()
            st.sidebar.download_button(
                label="📥 Download Source Code",
                data=zip_data,
                file_name="ai_stock_predictor_source.zip",
                mime="application/zip",
                help="Download all source code files in a zip archive"
            )
        except Exception as e:
            st.sidebar.error(f"Source code download error: {str(e)}")
    
    # Main content area
    if predict_button:
        if not symbol:
            st.error("Please enter a stock symbol!")
            return
        
        # Initialize components
        data_fetcher = StockDataFetcher()
        technical_analyzer = TechnicalAnalyzer()
        pattern_detector = PatternDetector()
        ml_predictor = MLPredictor(model_type=model_type.lower().replace(" ", "_"))
        visualizer = ChartVisualizer()
        explainer = ExplanationEngine()
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(f"📊 Analysis for {symbol.upper()}")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Fetch data
                status_text.text("🔄 Fetching stock data...")
                progress_bar.progress(20)
                df = data_fetcher.get_stock_data(symbol)
                
                if df is None or df.empty:
                    st.error("Failed to fetch stock data. Please check the symbol and try again.")
                    return
                
                # Step 2: Technical analysis
                status_text.text("📈 Calculating technical indicators...")
                progress_bar.progress(40)
                df_with_indicators = technical_analyzer.add_all_indicators(df)
                
                # Step 3: Pattern detection
                status_text.text("🔍 Detecting candlestick patterns...")
                progress_bar.progress(60)
                patterns = pattern_detector.detect_patterns(df_with_indicators)
                
                # Step 4: ML prediction
                status_text.text("🧠 Making AI prediction...")
                progress_bar.progress(80)
                prediction_result = ml_predictor.predict(df_with_indicators, timeframe)
                
                # Step 5: Generate explanation
                status_text.text("💭 Generating explanation...")
                progress_bar.progress(100)
                explanation = explainer.generate_explanation(
                    prediction_result, df_with_indicators, patterns, education_mode
                )
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Store data in session state for quick stats
                st.session_state.last_df = df_with_indicators
                
                # Store data in database if connected
                if st.session_state.get('db_initialized', False):
                    try:
                        db = get_database_manager()
                        db.store_stock_data(symbol, df)
                        db.store_technical_indicators(symbol, df_with_indicators)
                        db.store_prediction(symbol, prediction_result, timeframe, model_type.lower().replace(" ", "_"))
                        db.store_detected_patterns(symbol, patterns)
                    except Exception as e:
                        st.warning(f"Data storage warning: {str(e)}")
                
                # Display results
                display_prediction_results(prediction_result, explanation, confidence_threshold)
                
                # Display chart with fallback handling
                st.subheader("📈 Interactive Chart")
                try:
                    chart = visualizer.create_candlestick_chart(df_with_indicators, symbol, chart_type)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.error("Chart display failed. Trying line chart...")
                        chart = visualizer.create_candlestick_chart(df_with_indicators, symbol, "Line")
                        if chart is not None:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.error("Unable to display chart")
                except Exception as chart_error:
                    st.warning(f"Chart display issue: {str(chart_error)}")
                    st.info("Displaying simplified line chart...")
                    try:
                        chart = visualizer.create_candlestick_chart(df_with_indicators, symbol, "Line")
                        st.plotly_chart(chart, use_container_width=True)
                    except Exception as fallback_error:
                        st.error(f"Chart fallback failed: {str(fallback_error)}")
                
                # Technical indicators summary
                display_technical_summary(df_with_indicators, patterns)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different stock symbol or check your internet connection.")
        
        with col2:
            st.header("🎯 Quick Stats")
            if st.session_state.get('last_df') is not None:
                display_quick_stats(st.session_state.last_df)
            else:
                st.info("📊 Stats will appear after prediction")
            
            # Show recent predictions from database
            if st.session_state.get('db_initialized', False):
                st.header("📈 Recent Predictions")
                try:
                    db = get_database_manager()
                    history = db.get_prediction_history(days=1)
                    
                    if history:
                        # Show last 5 predictions
                        for pred in history[:5]:
                            with st.container():
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.write(f"**{pred['symbol']}** - {pred['timeframe']}")
                                    st.caption(pred['timestamp'].strftime("%H:%M:%S"))
                                with col2:
                                    color = "🟢" if pred['action'] == 'BUY' else "🔴" if pred['action'] == 'SELL' else "🟡"
                                    st.write(f"{color} {pred['action']}")
                                with col3:
                                    st.write(f"{pred['confidence']:.1%}")
                    else:
                        st.info("No recent predictions found")
                except Exception as e:
                    st.error(f"History error: {str(e)}")
            
            if education_mode:
                st.header("🎓 Learning Center")
                display_educational_content()
    
    else:
        # Landing page content
        display_landing_page()

def display_prediction_results(prediction_result, explanation, confidence_threshold):
    """Display the prediction results with styling"""
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    action = prediction_result['action']
    
    # Only show prediction if confidence is above threshold
    if confidence >= confidence_threshold:
        # Color coding based on action
        if action == "BUY":
            css_class = "buy-signal"
            emoji = "🟢"
        elif action == "SELL":
            css_class = "sell-signal"
            emoji = "🔴"
        else:
            css_class = "hold-signal"
            emoji = "🟡"
        
        st.markdown(f"""
        <div class="prediction-box {css_class}">
            {emoji} Prediction: {prediction.upper()} | Action: {action} | Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation
        st.info(f"💡 **AI Analysis:** {explanation}")
        
    else:
        st.warning(f"⚠️ Low confidence prediction ({confidence:.1%}). Consider waiting for a clearer signal.")

def display_technical_summary(df, patterns):
    """Display technical indicators summary"""
    if df is None or df.empty:
        return
    
    st.subheader("📊 Technical Analysis Summary")
    
    # Get latest values
    latest = df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI", f"{latest.get('RSI', 0):.1f}", 
                 help="Relative Strength Index (70+ overbought, 30- oversold)")
        st.metric("Price", f"${latest.get('Close', 0):.2f}")
    
    with col2:
        st.metric("MACD", f"{latest.get('MACD', 0):.3f}",
                 help="Moving Average Convergence Divergence")
        st.metric("Volume", f"{latest.get('Volume', 0):,.0f}")
    
    with col3:
        st.metric("Bollinger %B", f"{latest.get('BB_percent', 0):.2f}",
                 help="Position within Bollinger Bands")
        st.metric("ATR", f"{latest.get('ATR', 0):.2f}",
                 help="Average True Range (volatility)")
    
    # Patterns detected
    if patterns:
        st.subheader("🕯️ Detected Patterns")
        for pattern in patterns[-5:]:  # Show last 5 patterns
            st.write(f"• {pattern}")

def display_quick_stats(df):
    """Display quick statistics"""
    if df is None or df.empty:
        st.info("📊 Stats will appear after prediction")
        return
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Price change
    price_change = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100
    
    st.metric(
        "Price Change",
        f"${price_change:+.2f}",
        f"{price_change_pct:+.2f}%"
    )
    
    # High/Low
    st.metric("Day High", f"${latest['High']:.2f}")
    st.metric("Day Low", f"${latest['Low']:.2f}")
    
    # Volatility indicator
    volatility = df['Close'].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.2f}%")

def display_educational_content():
    """Display educational content for beginners"""
    with st.expander("📚 What is RSI?"):
        st.write("""
        **Relative Strength Index (RSI)** measures the speed and change of price movements.
        - **Above 70**: Stock might be overbought (good time to sell)
        - **Below 30**: Stock might be oversold (good time to buy)
        - **Around 50**: Neutral territory
        """)
    
    with st.expander("📚 Understanding MACD"):
        st.write("""
        **MACD (Moving Average Convergence Divergence)** shows the relationship between two moving averages.
        - **MACD > 0**: Bullish momentum
        - **MACD < 0**: Bearish momentum
        - **MACD crossing above signal line**: Potential buy signal
        """)
    
    with st.expander("📚 Bollinger Bands"):
        st.write("""
        **Bollinger Bands** consist of a middle line (moving average) and two outer bands.
        - **Price near upper band**: Potentially overbought
        - **Price near lower band**: Potentially oversold
        - **Bands squeezing**: Low volatility, potential breakout coming
        """)
    
    with st.expander("💡 Trading Tips"):
        st.write("""
        **Remember:**
        - Never invest more than you can afford to lose
        - This is for educational purposes, not financial advice
        - Always do your own research
        - Consider multiple indicators before making decisions
        - Practice with paper trading first
        """)

def display_landing_page():
    """Display landing page when no prediction is made"""
    st.markdown("""
    ## Welcome to AI Stock Predictor! 🚀
    
    This advanced AI-powered tool helps you analyze stock movements using:
    
    ### 🧠 Machine Learning
    - **Random Forest**, **XGBoost**, and **SVM** models
    - Trained on historical data and technical indicators
    - Real-time predictions with confidence scores
    
    ### 📊 Technical Analysis
    - **RSI** (Relative Strength Index)
    - **MACD** (Moving Average Convergence Divergence)
    - **Bollinger Bands**
    - **EMA** (Exponential Moving Average)
    - **ATR** (Average True Range)
    
    ### 🕯️ Pattern Recognition
    - Candlestick pattern detection
    - Support and resistance levels
    - Trend analysis
    
    ### 🎯 Smart Recommendations
    - **BUY/SELL/HOLD** signals
    - Confidence-based filtering
    - Natural language explanations
    
    ---
    
    **⚠️ Disclaimer:** This tool is for educational purposes only. Always do your own research before making investment decisions.
    """)
    
    # Sample stocks to try
    st.subheader("🔥 Popular Stocks to Analyze")
    popular_stocks = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "NVDA", "META", "NFLX"]
    
    cols = st.columns(4)
    for i, stock in enumerate(popular_stocks):
        with cols[i % 4]:
            if st.button(f"📈 {stock}", key=f"stock_{stock}"):
                st.session_state.selected_stock = stock
                st.rerun()

if __name__ == "__main__":
    main()
