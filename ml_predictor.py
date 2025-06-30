import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.svm import SVC, SVR
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    """Machine Learning predictor for stock price movements"""
    
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.classification_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        # Try to load existing model
        self._load_model()
    
    def _prepare_features(self, df):
        """
        Prepare features for machine learning
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators
        
        Returns:
            pandas.DataFrame: Prepared features
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Select relevant features for ML
        feature_cols = [
            'RSI', 'MACD', 'MACD_signal', 'BB_percent', 'BB_width',
            'ATR', 'ADX', 'STOCH_k', 'STOCH_d', 'WILLIAMS_R',
            'MFI', 'CMF', 'OBV', 'Volume_change', 'Price_change',
            'HL_pct', 'Close_position', 'Momentum_5', 'Volatility'
        ]
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            return pd.DataFrame()
        
        features = df[available_cols].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        features = features.fillna(0)
        
        # Add lag features
        for col in ['Close', 'Volume', 'RSI']:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Add rolling statistics
        if 'Close' in df.columns:
            features['Close_rolling_mean_5'] = df['Close'].rolling(5).mean()
            features['Close_rolling_std_5'] = df['Close'].rolling(5).std()
            features['Close_rolling_mean_10'] = df['Close'].rolling(10).mean()
            features['Close_rolling_std_10'] = df['Close'].rolling(10).std()
        
        # Drop rows with NaN (from lag features)
        features = features.dropna()
        
        return features
    
    def _create_targets(self, df, timeframe="5min"):
        """
        Create target variables for prediction
        
        Args:
            df (pandas.DataFrame): OHLCV data
            timeframe (str): Prediction timeframe
        
        Returns:
            tuple: (classification_target, regression_target)
        """
        if df is None or df.empty:
            return None, None
        
        # Determine future periods based on timeframe
        future_periods = {
            "1min": 1,
            "5min": 5,
            "10min": 10
        }
        
        periods = future_periods.get(timeframe, 5)
        
        # Calculate future returns
        future_prices = df['Close'].shift(-periods)
        current_prices = df['Close']
        
        returns = (future_prices - current_prices) / current_prices
        
        # Classification target (UP/DOWN/SUSTAIN)
        threshold = 0.002  # 0.2% threshold

        # Explicitly set categories before assignment
        classification_target = pd.Series(index=df.index, dtype='category')
        classification_target = classification_target.cat.set_categories(['UP', 'DOWN', 'SUSTAIN'])
        classification_target[returns > threshold] = 'UP'
        classification_target[returns < -threshold] = 'DOWN'
        classification_target[(returns >= -threshold) & (returns <= threshold)] = 'SUSTAIN'

        # Regression target (actual return percentage)
        regression_target = returns
        
        return classification_target, regression_target
    
    def train_models(self, df, timeframe="5min"):
        """
        Train both classification and regression models
        
        Args:
            df (pandas.DataFrame): Training data with technical indicators
            timeframe (str): Prediction timeframe
        
        Returns:
            dict: Training results
        """
        try:
            # Prepare features and targets
            features = self._prepare_features(df)
            class_target, reg_target = self._create_targets(df, timeframe)
            
            if features.empty or class_target is None:
                return {"error": "Insufficient data for training"}
            
            # Align features and targets
            common_index = features.index.intersection(class_target.index)
            features = features.loc[common_index]
            class_target = class_target.loc[common_index]
            reg_target = reg_target.loc[common_index]
            
            # Remove NaN values
            mask = ~(class_target.isna() | reg_target.isna())
            features = features[mask]
            class_target = class_target[mask]
            reg_target = reg_target[mask]

            if len(features) < 50:
                return {"error": "Insufficient training data"}

            # Check for at least two unique classes
            unique_classes = class_target.dropna().unique()
            if len(unique_classes) < 2:
                return {"error": "Not enough class variety in data for training. The model needs at least two classes (e.g., UP and DOWN). Try a different symbol or timeframe."}

            # Store feature columns
            self.feature_columns = features.columns.tolist()
            
            # Split data
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                features, class_target, test_size=0.2, random_state=42, stratify=class_target
            )
            
            _, _, y_reg_train, y_reg_test = train_test_split(
                features, reg_target, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train classification model
            if self.model_type == "random_forest":
                self.classification_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            elif self.model_type == "xgboost":
                self.classification_model = xgb.XGBClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            elif self.model_type == "svm":
                self.classification_model = SVC(
                    probability=True, random_state=42
                )
            
            self.classification_model.fit(X_train_scaled, y_class_train)
            
            # Train regression model
            if self.model_type == "random_forest":
                self.regression_model = RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            elif self.model_type == "xgboost":
                self.regression_model = xgb.XGBRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            elif self.model_type == "svm":
                self.regression_model = SVR()
            
            self.regression_model.fit(X_train_scaled, y_reg_train)
            
            # Evaluate models
            class_pred = self.classification_model.predict(X_test_scaled)
            class_accuracy = accuracy_score(y_class_test, class_pred)
            
            reg_score = self.regression_model.score(X_test_scaled, y_reg_test)
            
            self.is_trained = True
            
            # Save models
            self._save_model()
            
            return {
                "classification_accuracy": class_accuracy,
                "regression_score": reg_score,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}
    
    def predict(self, df, timeframe="5min"):
        """
        Make predictions on new data
        
        Args:
            df (pandas.DataFrame): Data with technical indicators
            timeframe (str): Prediction timeframe
        
        Returns:
            dict: Prediction results
        """
        try:
            # If models are not trained, train them first
            if not self.is_trained or self.classification_model is None:
                training_result = self.train_models(df, timeframe)
                if "error" in training_result:
                    return {
                        "prediction": "N/A",
                        "confidence": 0.0,
                        "action": "HOLD",
                        "error": training_result["error"]
                    }
            # Prepare features
            features = self._prepare_features(df)
            if features.empty:
                return {
                    "prediction": "N/A",
                    "confidence": 0.0,
                    "action": "HOLD",
                    "error": "No features available for prediction"
                }
            # Use the last row for prediction
            latest_features = features.iloc[-1:][self.feature_columns]
            # Handle missing columns
            for col in self.feature_columns:
                if col not in latest_features.columns:
                    latest_features[col] = 0
            # Scale features
            latest_scaled = self.scaler.transform(latest_features)
            # Make predictions
            class_pred = self.classification_model.predict(latest_scaled)[0]
            class_proba = self.classification_model.predict_proba(latest_scaled)[0]
            reg_pred = self.regression_model.predict(latest_scaled)[0]
            # Get confidence (max probability)
            confidence = max(class_proba)
            # Determine action
            if class_pred == 'UP':
                action = 'BUY'
            elif class_pred == 'DOWN':
                action = 'SELL'
            else:
                action = 'HOLD'
            # Adjust action based on confidence
            if confidence < 0.6:
                action = 'HOLD'
            return {
                "prediction": class_pred,
                "confidence": confidence,
                "action": action,
                "expected_return": reg_pred,
                "timeframe": timeframe,
                "probabilities": {
                    "UP": class_proba[list(self.classification_model.classes_).index('UP')] if 'UP' in self.classification_model.classes_ else 0,
                    "DOWN": class_proba[list(self.classification_model.classes_).index('DOWN')] if 'DOWN' in self.classification_model.classes_ else 0,
                    "SUSTAIN": class_proba[list(self.classification_model.classes_).index('SUSTAIN')] if 'SUSTAIN' in self.classification_model.classes_ else 0
                }
            }
        except Exception as e:
            return {
                "prediction": "N/A",
                "confidence": 0.0,
                "action": "HOLD",
                "error": f"Prediction failed: {str(e)}"
            }
    
    def _save_model(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'classification_model': self.classification_model,
                'regression_model': self.regression_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            
            filename = f"ml_models_{self.model_type}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def _load_model(self):
        """Load trained models from disk"""
        try:
            filename = f"ml_models_{self.model_type}.pkl"
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.classification_model = model_data.get('classification_model')
                self.regression_model = model_data.get('regression_model')
                self.scaler = model_data.get('scaler', StandardScaler())
                self.feature_columns = model_data.get('feature_columns', [])
                self.model_type = model_data.get('model_type', self.model_type)
                self.is_trained = model_data.get('is_trained', False)
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_trained = False
