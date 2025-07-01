# AI Stock Predictor - System Architecture Documentation

## Overview

This is an AI-powered stock market prediction web application built with Streamlit that provides real-time stock analysis and predictions. The system uses machine learning models, technical analysis, and pattern recognition to predict short-term stock price movements (10 seconds to 10 minutes) and provides actionable trading recommendations with natural language explanations.

## System Architecture

### Frontend Architecture

- **Framework**: Streamlit web application framework
- **Visualization**: Interactive charts using Plotly for candlestick charts and technical indicators
- **UI/UX**: Dark theme with custom CSS styling for professional trading interface
- **Responsiveness**: Wide layout configuration with expandable sidebar for controls

### Backend Architecture

- **Modular Design**: Component-based architecture with separate modules for different functionalities
- **Data Processing Pipeline**: Sequential data flow from fetching → analysis → prediction → visualization
- **Model Management**: Pickle-based model persistence with automatic loading/creation
- **Real-time Processing**: Live data fetching and analysis capabilities
- **Database Integration**: PostgreSQL database for data persistence and historical analysis
- **Session Management**: User preferences and prediction history tracking

### Machine Learning Pipeline

- **Algorithms**: Multiple ML models including Random Forest, XGBoost, and SVM
- **Feature Engineering**: 19+ technical indicators as input features
- **Model Types**: Both classification (direction prediction) and regression (price target)
- **Training Strategy**: Automatic model training with feature scaling and validation

## Key Components

### 1. Data Layer (`data_fetcher.py`)

- **Purpose**: Fetches real-time stock data from Alpha Vantage API
- **Features**: Supports multiple timeframes (1min, 5min, 15min, 30min, 60min)
- **Fallback**: Demo data generation when API limits are reached
- **Error Handling**: Comprehensive error management for API failures

### 8. Database Layer (`database.py`)

- **Purpose**: PostgreSQL database integration for data persistence
- **Tables**: StockData, TechnicalIndicators, Predictions, DetectedPatterns, UserSessions
- **Features**: Historical data storage, prediction tracking, user session management
- **Analytics**: Database statistics and prediction history analysis

### 2. Technical Analysis Engine (`technical_analysis.py`)

- **Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, Williams %R, MFI, CMF, OBV
- **Categories**: Trend, momentum, volatility, and volume indicators
- **Library**: Uses the `ta` (Technical Analysis) library for reliable calculations
- **Data Cleaning**: Automatic handling of missing data and outliers

### 3. Pattern Recognition (`pattern_detector.py`)

- **Candlestick Patterns**: Hammer, Doji, Engulfing, Morning/Evening Star, etc.
- **Chart Patterns**: Support/Resistance, trend lines, breakouts
- **Technical Patterns**: Moving average crossovers, momentum divergences
- **Signal Generation**: Bullish/bearish pattern classification

### 4. Machine Learning Predictor (`ml_predictor.py`)

- **Model Selection**: Configurable ML algorithms (Random Forest, XGBoost, SVM)
- **Feature Set**: 19 engineered features from technical indicators
- **Dual Prediction**: Classification for direction + regression for price targets
- **Model Persistence**: Automatic saving/loading of trained models

### 5. Visualization Engine (`visualization.py`)

- **Chart Types**: Candlestick, line, and OHLC charts
- **Subplots**: Multi-panel layout with price, volume, RSI, and MACD
- **Interactivity**: Zoom, pan, hover tooltips, and indicator overlays
- **Real-time Updates**: Dynamic chart updates with new data

### 6. AI Explanation Engine (`explanation_engine.py`)

- **Natural Language**: Converts technical analysis into plain English explanations
- **Template System**: Context-aware explanation templates for different market conditions
- **Educational Mode**: Beginner-friendly explanations with learning content
- **Confidence Scoring**: Explanation quality based on signal strength

### 7. Utility Functions (`utils.py`)

- **Model Management**: Loading, saving, and validation of ML models
- **Data Utilities**: Time formatting, data validation, and helper functions
- **Configuration**: Environment variable management and settings

## Data Flow

1. **Data Acquisition**: Stock symbol input → Alpha Vantage API call → Raw OHLCV data
2. **Technical Analysis**: Raw data → Technical indicators calculation → Enhanced dataset
3. **Pattern Detection**: Enhanced data → Pattern recognition → Signal identification
4. **ML Prediction**: Features → Trained model → Prediction + confidence score
5. **Explanation Generation**: Prediction + indicators → Natural language explanation
6. **Visualization**: All data → Interactive charts → User display
7. **Action Recommendation**: Prediction + patterns → BUY/SELL/HOLD recommendation

## External Dependencies

### Core Libraries

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **ta**: Technical analysis indicators
- **requests**: HTTP API calls

### API Services

- **Alpha Vantage**: Free stock market data provider
- **Environment Variables**: API key management through `ALPHA_VANTAGE_API_KEY`

### Development Tools

- **pickle**: Model serialization
- **warnings**: Error suppression for cleaner output
- **datetime/time**: Time handling and formatting

## Deployment Strategy

### Platform Configuration

- **Target Platform**: Replit with automatic scaling deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: Streamlit server on port 5000
- **Package Management**: UV package manager for dependency resolution

### Deployment Process

1. **Dependency Installation**: Automatic installation via UV package manager
2. **Environment Setup**: Nix-based environment with required system packages
3. **Application Launch**: Streamlit server with custom port configuration
4. **Autoscaling**: Automatic resource scaling based on demand

### Performance Optimizations

- **Model Caching**: Pickle-based model persistence to avoid retraining
- **Data Caching**: Streamlit's caching mechanisms for API responses
- **Lazy Loading**: Components loaded on-demand for faster startup
- **Error Resilience**: Graceful degradation when services are unavailable

## Changelog

- June 27, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.

## Main thing you should know

first if you can enter these line in CMD and then run the wepapp using streamlit.

```bash
set DATABASE_URL=sqlite:///mydb.sqlite3
```
