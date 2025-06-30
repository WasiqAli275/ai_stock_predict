# AI Stock Predictor - System Architecture Documentation

## Overview

This is an AI-powered stock market prediction web application built with Streamlit. It provides real-time stock analysis and predictions using machine learning models, technical analysis, and pattern recognition. The app predicts short-term stock price movements (1min, 5min, 10min) and provides actionable trading recommendations with natural language explanations.

## Features

- **Real-time Stock Data**: Fetches live stock data from Alpha Vantage API.
- **Multiple ML Models**: Supports Random Forest, XGBoost, and SVM for predictions.
- **Technical Analysis**: Calculates 19+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.).
- **Pattern Recognition**: Detects candlestick and chart patterns (Hammer, Doji, Engulfing, etc.).
- **Interactive Charts**: Uses Plotly for candlestick, line, and OHLC charts with technical overlays.
- **Natural Language Explanations**: Converts technical signals into plain English for all users.
- **Educational Mode**: Beginner-friendly explanations and learning content.
- **Database Integration**: Stores data, predictions, and user sessions in PostgreSQL.
- **Downloadable Markdown**: Users can download the system documentation as a markdown file.
- **Error Handling**: Graceful error messages for API/data/model issues.
- **Custom Styling**: Modern UI with custom CSS, emoji support, and responsive layout.

## System Architecture

### Frontend

- **Framework**: Streamlit
- **Visualization**: Plotly for interactive charts
- **UI/UX**: Custom CSS, emoji icons, wide layout, sidebar controls
- **Download**: Markdown file download button in sidebar

### Backend

- **Data Fetching**: Alpha Vantage API (with API key)
- **Technical Analysis**: `ta` library for indicators
- **Pattern Detection**: Custom pattern recognition module
- **ML Prediction**: Random Forest, XGBoost, SVM (classification + regression)
- **Database**: PostgreSQL for persistence and analytics
- **Model Management**: Pickle-based model saving/loading

## Data Flow

1. **User Input**: Stock symbol, timeframe, model selection
2. **Data Fetching**: Real-time data from Alpha Vantage
3. **Technical Analysis**: Indicators calculated and added to data
4. **Pattern Detection**: Candlestick and chart patterns identified
5. **ML Prediction**: Model predicts direction and confidence
6. **Explanation**: AI generates natural language summary
7. **Visualization**: Interactive chart and stats displayed
8. **Database Storage**: Data and predictions saved for analytics
9. **Download**: User can download documentation as markdown

## Key Components

- `app.py`: Main Streamlit app, UI, and workflow
- `data_fetcher.py`: Fetches and validates stock data
- `technical_analysis.py`: Adds technical indicators
- `pattern_detector.py`: Detects chart/candlestick patterns
- `ml_predictor.py`: ML model training, prediction, and management
- `visualization.py`: Chart creation and display
- `explanation_engine.py`: Generates natural language explanations
- `database.py`: Handles database operations
- `utils.py`: Helper functions and model utilities

## Usage

1. **Set Environment Variables**:
   - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key
   - `DATABASE_URL`: PostgreSQL connection string (e.g., `postgresql://user:pass@host:port/db`)
2. **Install Requirements**:
   - All dependencies listed in `requirements.md`
3. **Run the App**:
   - `streamlit run app.py`
4. **Interact**:
   - Enter a stock symbol (e.g., AAPL, MSFT, TSLA)
   - Select timeframe and model
   - Click Predict to see analysis, chart, and explanation
   - Download documentation from the sidebar

## Error Handling

- **API Errors**: User-friendly messages if symbol is invalid or API limit is reached
- **Model Errors**: Clear feedback if not enough data or class variety for training
- **Chart Errors**: Fallback to line chart if candlestick fails
- **Download Errors**: Message if markdown file is missing

## Customization

- **Styling**: Custom CSS for headers, prediction boxes, and emoji icons
- **Educational Content**: Expanders in sidebar for RSI, MACD, Bollinger Bands, and trading tips
- **Popular Stocks**: Quick buttons for popular US stocks

## Deployment

- **Platform**: Replit (or any Python 3.11+ environment)
- **Port**: 5000 (configurable)
- **Scaling**: Autoscaling supported on Replit
- **Persistence**: Database and model files saved between sessions

## Changelog

- June 27, 2025: Initial setup
- June 30, 2025: Improved error handling, markdown download, and UI enhancements

## User Preferences

Preferred communication style: Simple, everyday language.
