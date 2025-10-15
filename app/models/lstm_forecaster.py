import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class LSTMForecaster:
    """Advanced LSTM model for financial time series forecasting"""
    
    def __init__(self, sequence_length: int = 60, features_to_use: List[str] = None):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.training_history = None
        self.model_performance = {}
        
        # Default features to use
        self.features_to_use = features_to_use or [
            'Close', 'Volume', 'High', 'Low', 'Open'
        ]
        
        # Model architecture parameters
        self.model_config = {
            'lstm_units': [100, 50, 25],
            'dropout_rate': 0.3,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators without pandas-ta"""

        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_lower'] = rolling_mean - (rolling_std * 2)

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        return df
    
    
    def _fetch_and_prepare_data(self, symbol: str, period: str = "3y") -> pd.DataFrame:
        """Fetch stock data and add technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Add technical indicators for better predictions
            df['RSI'] = df.rsi(df['Close'], length=14)
            df['MACD'] = df.macd(df['Close'])['MACD_12_26_9']
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = df.bbands(df['Close'], length=20).iloc[:, [0, 1, 2]].T.values
            df['SMA_20'] = df.sma(df['Close'], length=20)
            df['SMA_50'] = df.sma(df['Close'], length=50)
            df['EMA_12'] = df.ema(df['Close'], length=12)
            df['ATR'] = df.atr(df['High'], df['Low'], df['Close'], length=14)
            df['Volume_SMA'] = df.sma(df['Volume'], length=20)
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Close'].rolling(window=20).std()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
            
            # Extended feature list
            self.features_to_use = [
                'Close', 'Volume', 'High', 'Low', 'Open', 'RSI', 'MACD', 
                'BB_upper', 'BB_lower', 'SMA_20', 'SMA_50', 'EMA_12', 
                'ATR', 'Price_Change', 'Volatility', 'Volume_Ratio', 'High_Low_Ratio'
            ]
            
            return df.dropna()
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _prepare_lstm_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        
        # Select and scale features
        feature_data = data[self.features_to_use].values
        scaled_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        
        # Create sequences
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict Close price (index 0)
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the LSTM model architecture"""
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.model_config['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=0.2
        ))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.model_config['lstm_units'][1],
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=0.2
        ))
        model.add(BatchNormalization())
        
        # Third LSTM layer
        model.add(LSTM(
            units=self.model_config['lstm_units'][2],
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=0.2
        ))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.model_config['dropout_rate']))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))  # Output layer for price prediction
        
        # Compile model
        optimizer = Adam(learning_rate=self.model_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, symbol: str, validation_split: float = 0.2) -> Dict:
        """Train the LSTM model"""
        
        print(f"Training LSTM model for {symbol}...")
        
        # Fetch and prepare data
        df = self._fetch_and_prepare_data(symbol)
        X, y = self._prepare_lstm_data(df)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Features used: {self.features_to_use}")
        
        # Split data (keep time series order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X_train, y_train,
            batch_size=self.model_config['batch_size'],
            epochs=self.model_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        # Evaluate model performance
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Inverse transform predictions for evaluation
        train_pred_inv = self._inverse_transform_predictions(train_pred)
        val_pred_inv = self._inverse_transform_predictions(val_pred)
        y_train_inv = self._inverse_transform_predictions(y_train.reshape(-1, 1))
        y_val_inv = self._inverse_transform_predictions(y_val.reshape(-1, 1))
        
        self.model_performance = {
            'train_mae': mean_absolute_error(y_train_inv, train_pred_inv),
            'train_mse': mean_squared_error(y_train_inv, train_pred_inv),
            'train_r2': r2_score(y_train_inv, train_pred_inv),
            'val_mae': mean_absolute_error(y_val_inv, val_pred_inv),
            'val_mse': mean_squared_error(y_val_inv, val_pred_inv),
            'val_r2': r2_score(y_val_inv, val_pred_inv),
            'final_train_loss': self.training_history.history['loss'][-1],
            'final_val_loss': self.training_history.history['val_loss'][-1]
        }
        
        self.is_trained = True
        
        print(f"Training completed!")
        print(f"Validation R² Score: {self.model_performance['val_r2']:.4f}")
        print(f"Validation MAE: {self.model_performance['val_mae']:.4f}")
        
        return self.model_performance
    
    def _inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions to original scale"""
        
        # Create dummy array with same shape as original features
        dummy_array = np.zeros((predictions.shape[0], len(self.features_to_use)))
        dummy_array[:, 0] = predictions.flatten()  # Close price is at index 0
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_array)
        
        return inverse_transformed[:, 0]  # Return only Close price predictions
    
    def predict(self, symbol: str, days_ahead: int = 7) -> Dict:
        """Make predictions for specified number of days ahead"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Fetch latest data
        df = self._fetch_and_prepare_data(symbol, period="1y")
        
        # Prepare data for prediction
        feature_data = df[self.features_to_use].values
        scaled_data = self.scaler.transform(feature_data)
        
        # Get last sequence for prediction
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features_to_use))
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        # Generate predictions for multiple days
        for day in range(days_ahead):
            # Predict next day
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            # Create new features (simplified approach - in production, you'd want more sophisticated feature engineering)
            new_features = current_sequence[0, -1].copy()
            new_features[0] = next_pred  # Update Close price
            
            # Shift sequence and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_features
        
        # Convert predictions back to original scale
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_inv = self._inverse_transform_predictions(predictions_array)
        
        # Calculate confidence based on model performance and prediction consistency
        base_confidence = max(0.1, min(0.9, self.model_performance.get('val_r2', 0.5)))
        
        # Adjust confidence based on prediction volatility
        if len(predictions_inv) > 1:
            pred_volatility = np.std(predictions_inv) / np.mean(predictions_inv)
            volatility_penalty = min(pred_volatility * 2, 0.3)
            confidence = base_confidence * (1 - volatility_penalty)
        else:
            confidence = base_confidence
        
        current_price = float(df['Close'].iloc[-1])
        final_prediction = float(predictions_inv[-1])
        
        return {
            'predictions': predictions_inv.tolist(),
            'final_prediction': final_prediction,
            'current_price': current_price,
            'predicted_change_pct': ((final_prediction - current_price) / current_price) * 100,
            'confidence': float(confidence),
            'days_ahead': days_ahead,
            'model_performance': self.model_performance
        }
    
    def get_feature_importance(self, symbol: str) -> Dict:
        """Analyze feature importance using gradient-based method"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        # Fetch data
        df = self._fetch_and_prepare_data(symbol, period="6m")
        feature_data = df[self.features_to_use].values
        scaled_data = self.scaler.transform(feature_data)
        
        # Get a sample of recent sequences
        sample_size = min(100, len(scaled_data) - self.sequence_length)
        sample_sequences = []
        
        for i in range(sample_size):
            idx = len(scaled_data) - self.sequence_length - sample_size + i
            sample_sequences.append(scaled_data[idx:idx+self.sequence_length])
        
        sample_sequences = np.array(sample_sequences)
        
        # Calculate gradients for each feature
        with tf.GradientTape() as tape:
            inputs = tf.Variable(sample_sequences, dtype=tf.float32)
            tape.watch(inputs)
            predictions = self.model(inputs)
            loss = tf.reduce_mean(predictions)
        
        gradients = tape.gradient(loss, inputs)
        
        # Calculate feature importance scores
        feature_importance = {}
        grad_abs_mean = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        for i, feature_name in enumerate(self.features_to_use):
            feature_importance[feature_name] = float(grad_abs_mean[i])
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return feature_importance
    
    def save_model(self, filepath: str):
        """Save the trained LSTM model"""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model




        model_path = f"{filepath}_model.h5"
        self.model.save(model_path)
        
        # Save scaler and other components
        components = {
            'scaler': self.scaler,
            'features_to_use': self.features_to_use,
            'sequence_length': self.sequence_length,
            'model_config': self.model_config,
            'model_performance': self.model_performance
        }
        
        joblib.dump(components, f"{filepath}_components.pkl")
        
        print(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained LSTM model"""
        
        # Load model
        model_path = f"{filepath}_model.h5"
        self.model = load_model(model_path)
        
        # Load components
        components = joblib.load(f"{filepath}_components.pkl")
        
        self.scaler = components['scaler']
        self.features_to_use = components['features_to_use']
        self.sequence_length = components['sequence_length']
        self.model_config = components['model_config']
        self.model_performance = components['model_performance']
        self.is_trained = True
        
        print(f"LSTM model loaded from {filepath}")