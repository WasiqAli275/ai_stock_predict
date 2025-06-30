import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class for ORM models
Base = declarative_base()

class StockData(Base):
    """Table for storing historical stock data"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    interval = Column(String(10), nullable=False)  # 1min, 5min, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

class TechnicalIndicators(Base):
    """Table for storing calculated technical indicators"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_percent = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    sma_20 = Column(Float)
    atr = Column(Float)
    adx = Column(Float)
    volume_sma = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Predictions(Base):
    """Table for storing ML predictions and results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    prediction_time = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1min, 5min, 10min
    model_type = Column(String(20), nullable=False)  # random_forest, xgboost, svm
    prediction = Column(String(10), nullable=False)  # UP, DOWN, SUSTAIN
    confidence = Column(Float, nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    expected_return = Column(Float)
    probabilities = Column(Text)  # JSON string of probabilities
    actual_outcome = Column(String(10))  # For backtesting
    accuracy = Column(Boolean)  # True if prediction was correct
    created_at = Column(DateTime, default=datetime.utcnow)

class DetectedPatterns(Base):
    """Table for storing detected candlestick and chart patterns"""
    __tablename__ = 'detected_patterns'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False)  # hammer, doji, engulfing, etc.
    pattern_description = Column(Text, nullable=False)
    strength = Column(String(10))  # weak, moderate, strong
    bullish_bearish = Column(String(10))  # bullish, bearish, neutral
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSessions(Base):
    """Table for tracking user sessions and preferences"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    last_symbol = Column(String(10))
    preferred_timeframe = Column(String(10))
    preferred_model = Column(String(20))
    education_mode = Column(Boolean, default=True)
    total_predictions = Column(Integer, default=0)
    last_activity = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def store_stock_data(self, symbol, df, interval="1min"):
        """
        Store stock OHLCV data in database
        
        Args:
            symbol (str): Stock symbol
            df (pandas.DataFrame): OHLCV data with datetime index
            interval (str): Data interval
        """
        try:
            with self.get_session() as session:
                for timestamp, row in df.iterrows():
                    # Check if data already exists
                    existing = session.query(StockData).filter_by(
                        symbol=symbol,
                        timestamp=timestamp,
                        interval=interval
                    ).first()
                    
                    if not existing:
                        stock_data = StockData(
                            symbol=symbol,
                            timestamp=timestamp,
                            open_price=float(row['Open']),
                            high_price=float(row['High']),
                            low_price=float(row['Low']),
                            close_price=float(row['Close']),
                            volume=int(row['Volume']),
                            interval=interval
                        )
                        session.add(stock_data)
                
                session.commit()
                logger.info(f"Stored {len(df)} stock data points for {symbol}")
                
        except Exception as e:
            logger.error(f"Error storing stock data: {str(e)}")
            raise
    
    def store_technical_indicators(self, symbol, df):
        """
        Store technical indicators in database
        
        Args:
            symbol (str): Stock symbol
            df (pandas.DataFrame): Data with technical indicators
        """
        try:
            with self.get_session() as session:
                for timestamp, row in df.iterrows():
                    # Check if indicators already exist
                    existing = session.query(TechnicalIndicators).filter_by(
                        symbol=symbol,
                        timestamp=timestamp
                    ).first()
                    
                    if not existing:
                        indicators = TechnicalIndicators(
                            symbol=symbol,
                            timestamp=timestamp,
                            rsi=float(row.get('RSI', 0)) if pd.notna(row.get('RSI')) else None,
                            macd=float(row.get('MACD', 0)) if pd.notna(row.get('MACD')) else None,
                            macd_signal=float(row.get('MACD_signal', 0)) if pd.notna(row.get('MACD_signal')) else None,
                            bb_upper=float(row.get('BB_upper', 0)) if pd.notna(row.get('BB_upper')) else None,
                            bb_middle=float(row.get('BB_middle', 0)) if pd.notna(row.get('BB_middle')) else None,
                            bb_lower=float(row.get('BB_lower', 0)) if pd.notna(row.get('BB_lower')) else None,
                            bb_percent=float(row.get('BB_percent', 0)) if pd.notna(row.get('BB_percent')) else None,
                            ema_12=float(row.get('EMA_12', 0)) if pd.notna(row.get('EMA_12')) else None,
                            ema_26=float(row.get('EMA_26', 0)) if pd.notna(row.get('EMA_26')) else None,
                            sma_20=float(row.get('SMA_20', 0)) if pd.notna(row.get('SMA_20')) else None,
                            atr=float(row.get('ATR', 0)) if pd.notna(row.get('ATR')) else None,
                            adx=float(row.get('ADX', 0)) if pd.notna(row.get('ADX')) else None,
                            volume_sma=float(row.get('Volume_SMA', 0)) if pd.notna(row.get('Volume_SMA')) else None
                        )
                        session.add(indicators)
                
                session.commit()
                logger.info(f"Stored technical indicators for {symbol}")
                
        except Exception as e:
            logger.error(f"Error storing technical indicators: {str(e)}")
    
    def store_prediction(self, symbol, prediction_result, timeframe, model_type):
        """
        Store ML prediction results
        
        Args:
            symbol (str): Stock symbol
            prediction_result (dict): Prediction results
            timeframe (str): Prediction timeframe
            model_type (str): ML model type used
        """
        try:
            with self.get_session() as session:
                prediction = Predictions(
                    symbol=symbol,
                    prediction_time=datetime.utcnow(),
                    timeframe=timeframe,
                    model_type=model_type,
                    prediction=prediction_result.get('prediction', 'SUSTAIN'),
                    confidence=float(prediction_result.get('confidence', 0.5)),
                    action=prediction_result.get('action', 'HOLD'),
                    expected_return=float(prediction_result.get('expected_return', 0)),
                    probabilities=json.dumps(prediction_result.get('probabilities', {}))
                )
                session.add(prediction)
                session.commit()
                logger.info(f"Stored prediction for {symbol}")
                
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
    
    def store_detected_patterns(self, symbol, patterns):
        """
        Store detected candlestick patterns
        
        Args:
            symbol (str): Stock symbol
            patterns (list): List of detected patterns
        """
        try:
            with self.get_session() as session:
                current_time = datetime.utcnow()
                
                for pattern in patterns:
                    # Extract pattern information
                    pattern_str = str(pattern).lower()
                    
                    # Determine pattern type and sentiment
                    if any(word in pattern_str for word in ['bullish', 'hammer', 'morning', 'piercing']):
                        sentiment = 'bullish'
                    elif any(word in pattern_str for word in ['bearish', 'shooting', 'evening', 'dark']):
                        sentiment = 'bearish'
                    else:
                        sentiment = 'neutral'
                    
                    # Determine pattern type
                    if 'doji' in pattern_str:
                        pattern_type = 'doji'
                    elif 'hammer' in pattern_str:
                        pattern_type = 'hammer'
                    elif 'engulfing' in pattern_str:
                        pattern_type = 'engulfing'
                    elif 'star' in pattern_str:
                        pattern_type = 'star'
                    elif 'macd' in pattern_str:
                        pattern_type = 'macd_signal'
                    elif 'rsi' in pattern_str:
                        pattern_type = 'rsi_signal'
                    else:
                        pattern_type = 'other'
                    
                    detected_pattern = DetectedPatterns(
                        symbol=symbol,
                        timestamp=current_time,
                        pattern_type=pattern_type,
                        pattern_description=str(pattern),
                        strength='moderate',  # Default strength
                        bullish_bearish=sentiment
                    )
                    session.add(detected_pattern)
                
                session.commit()
                logger.info(f"Stored {len(patterns)} patterns for {symbol}")
                
        except Exception as e:
            logger.error(f"Error storing patterns: {str(e)}")
    
    def get_recent_stock_data(self, symbol, hours=24, interval="1min"):
        """
        Retrieve recent stock data from database
        
        Args:
            symbol (str): Stock symbol
            hours (int): Hours of data to retrieve
            interval (str): Data interval
        
        Returns:
            pandas.DataFrame: Stock data
        """
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                query = session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.interval == interval,
                    StockData.timestamp >= cutoff_time
                ).order_by(StockData.timestamp)
                
                results = query.all()
                
                if not results:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in results:
                    data.append({
                        'Open': row.open_price,
                        'High': row.high_price,
                        'Low': row.low_price,
                        'Close': row.close_price,
                        'Volume': row.volume,
                        'timestamp': row.timestamp
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving stock data: {str(e)}")
            return pd.DataFrame()
    
    def get_prediction_history(self, symbol=None, days=7):
        """
        Get prediction history for analysis
        
        Args:
            symbol (str, optional): Stock symbol filter
            days (int): Days of history to retrieve
        
        Returns:
            list: Prediction history
        """
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                
                query = session.query(Predictions).filter(
                    Predictions.prediction_time >= cutoff_time
                )
                
                if symbol:
                    query = query.filter(Predictions.symbol == symbol)
                
                query = query.order_by(Predictions.prediction_time.desc())
                results = query.all()
                
                history = []
                for pred in results:
                    history.append({
                        'symbol': pred.symbol,
                        'timestamp': pred.prediction_time,
                        'prediction': pred.prediction,
                        'confidence': pred.confidence,
                        'action': pred.action,
                        'model_type': pred.model_type,
                        'timeframe': pred.timeframe
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error retrieving prediction history: {str(e)}")
            return []
    
    def update_session_preferences(self, session_id, preferences):
        """
        Update user session preferences
        
        Args:
            session_id (str): Session identifier
            preferences (dict): User preferences
        """
        try:
            with self.get_session() as session:
                user_session = session.query(UserSessions).filter_by(
                    session_id=session_id
                ).first()
                
                if not user_session:
                    user_session = UserSessions(session_id=session_id)
                    session.add(user_session)
                
                # Update preferences
                if 'last_symbol' in preferences:
                    user_session.last_symbol = preferences['last_symbol']
                if 'preferred_timeframe' in preferences:
                    user_session.preferred_timeframe = preferences['preferred_timeframe']
                if 'preferred_model' in preferences:
                    user_session.preferred_model = preferences['preferred_model']
                if 'education_mode' in preferences:
                    user_session.education_mode = preferences['education_mode']
                if 'total_predictions' in preferences:
                    user_session.total_predictions = preferences['total_predictions']
                
                # Update last activity time using update() method
                session.query(UserSessions).filter_by(
                    session_id=session_id
                ).update({'last_activity': datetime.utcnow()})
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Error updating session preferences: {str(e)}")
    
    def get_database_stats(self):
        """
        Get database statistics for monitoring
        
        Returns:
            dict: Database statistics
        """
        try:
            with self.get_session() as session:
                stats = {}
                
                # Count records in each table
                stats['stock_data_count'] = session.query(StockData).count()
                stats['predictions_count'] = session.query(Predictions).count()
                stats['patterns_count'] = session.query(DetectedPatterns).count()
                stats['indicators_count'] = session.query(TechnicalIndicators).count()
                stats['sessions_count'] = session.query(UserSessions).count()
                
                # Get unique symbols tracked
                symbols_query = session.query(StockData.symbol).distinct()
                stats['unique_symbols'] = [row[0] for row in symbols_query.all()]
                
                # Get latest data timestamp
                latest_data = session.query(StockData.timestamp).order_by(
                    StockData.timestamp.desc()
                ).first()
                stats['latest_data_time'] = latest_data[0] if latest_data else None
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}

# Global database manager instance
db_manager = None

def get_database_manager():
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def init_database():
    """Initialize database connection"""
    try:
        get_database_manager()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False