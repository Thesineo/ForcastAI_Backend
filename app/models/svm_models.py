import numpy as np
import pandas as pd
import yfinance as yf
import requests
import openai
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from textblob import TextBlob
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import joblib
import os
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class PredictionResult:
    """Structured prediction result with confidence metrics"""
    predicted_price: float
    confidence_score: float
    prediction_range: Tuple[float, float]
    risk_level: str
    explanation: str
    technical_indicators: Dict
    news_sentiment: Dict
    market_context: Dict
    timeframe: int

class EnhancedSVMPredictor:
    """Advanced SVM predictor with comprehensive market analysis"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_names = []
        self.is_trained = False
        self.model_performance = {}
        
        # Model parameters optimized for financial data
        self.model_params = {
            'svr__kernel': 'rbf',
            'svr__C': 100,
            'svr__gamma': 'scale',
            'svr__epsilon': 0.01
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
            """Calculate technical indicators manually without pandas-ta"""
    
            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
    
            # MACD calculation
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
    
            # Bollinger Bands
            rolling_mean = df['Close'].rolling(window=20).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = rolling_mean + (rolling_std * 2)
            df['BB_middle'] = rolling_mean
            df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
            # Exponential Moving Average
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
    
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
            # Average True Range (ATR)
            df['prev_close'] = df['Close'].shift(1)
            df['high_low'] = df['High'] - df['Low']
            df['high_close'] = np.abs(df['High'] - df['prev_close'])
            df['low_close'] = np.abs(df['Low'] - df['prev_close'])
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['ATR'] = df['true_range'].rolling(window=14).mean()
    
            # ADX (simplified version)
            df['high_diff'] = df['High'].diff()
            df['low_diff'] = df['Low'].diff()
            df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
            df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)
            df['plus_di'] = (df['plus_dm'].rolling(window=14).mean() / df['ATR']) * 100
            df['minus_di'] = (df['minus_dm'].rolling(window=14).mean() / df['ATR']) * 100
            df['dx'] = (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
            df['ADX'] = df['dx'].rolling(window=14).mean()
    
            # CCI (Commodity Channel Index)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
            # Williams %R
            df['Williams_R'] = ((df['High'].rolling(window=14).max() - df['Close']) / 
                       (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * -100
    
            # Stochastic %K
            df['lowest_low'] = df['Low'].rolling(window=14).min()
            df['highest_high'] = df['High'].rolling(window=14).max()
            df['Stoch_K'] = ((df['Close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])) * 100
    
            return df
    
    
    def _fetch_comprehensive_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch comprehensive market data including technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Add technical indicators using manual calculations
            df = self._calculate_technical_indicators(df)
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['Volatility'] = df['Close'].rolling(window=20).std()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
            
            return df.dropna()
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _get_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict:
        """Fetch and analyze news sentiment for the symbol"""
        try:
            # Using NewsAPI (you'll need to sign up for a free API key)
            news_api_key = os.getenv('NEWS_API_KEY')
            if not news_api_key:
                return {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': 'neutral'}
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock OR {symbol} earnings OR {symbol} finance",
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': news_api_key,
                'pageSize': 50
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                if not articles:
                    return {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': 'neutral'}
                
                sentiments = []
                for article in articles[:20]:  # Analyze top 20 articles
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}"
                    
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                
                avg_sentiment = np.mean(sentiments)
                sentiment_label = 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
                
                return {
                    'sentiment_score': float(avg_sentiment),
                    'news_count': len(articles),
                    'sentiment_label': sentiment_label,
                    'confidence': min(len(articles) / 10, 1.0)
                }
            
            return {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': 'neutral'}
            
        except Exception as e:
            print(f"Error fetching news sentiment: {str(e)}")
            return {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': 'neutral'}
    
    def _make_advanced_features(self, df: pd.DataFrame, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create advanced feature set including technical indicators and market conditions"""
        
        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 
            'SMA_20', 'SMA_50', 'EMA_12', 'ATR', 'ADX', 'CCI', 'Williams_R', 
            'Stoch_K', 'Price_Change', 'Volatility', 'Volume_Ratio', 'High_Low_Ratio'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        
        X, y = [], []
        feature_names = []
        
        # Create feature names for the lookback window
        for col in available_columns:
            for i in range(lookback):
                feature_names.append(f"{col}_lag_{i+1}")
        
        data_matrix = df[available_columns].values
        target = df['Close'].values
        
        for i in range(lookback, len(data_matrix)):
            # Create feature vector with lookback window
            window_features = []
            for j in range(lookback):
                window_features.extend(data_matrix[i - lookback + j])
            
            X.append(window_features)
            y.append(target[i])
        
        return np.array(X, dtype=float), np.array(y, dtype=float), feature_names
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize SVM hyperparameters using time series cross-validation"""
        
        param_grid = {
            'svr__C': [10, 50, 100, 200],
            'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svr__epsilon': [0.01, 0.1, 0.2]
        }
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(kernel='rbf'))
        ])
        
        # Use TimeSeriesSplit for proper cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_
    
    def train_model(self, symbol: str, optimize_params: bool = True) -> Dict:
        """Train the enhanced SVM model with comprehensive feature engineering"""
        
        print(f"Training enhanced SVM model for {symbol}...")
        
        # Fetch comprehensive data
        df = self._fetch_comprehensive_data(symbol)
        
        # Create advanced features
        X, y, feature_names = self._make_advanced_features(df)
        self.feature_names = feature_names
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Optimize hyperparameters if requested
        if optimize_params:
            print("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train, y_train)
            self.model_params.update(best_params)
        
        # Train final model
        self.model = Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(**{k.replace('svr__', ''): v for k, v in self.model_params.items() if k.startswith('svr__')}))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.model_performance = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        self.is_trained = True
        
        print(f"Model trained successfully!")
        print(f"Test R² Score: {self.model_performance['test_r2']:.4f}")
        print(f"Test MAE: {self.model_performance['test_mae']:.4f}")
        
        return self.model_performance
    
    def _calculate_confidence_score(self, symbol: str, prediction: float, current_price: float) -> Tuple[float, str]:
        """Calculate confidence score based on multiple factors"""
        
        # Base confidence from model performance
        base_confidence = max(0.1, min(0.9, self.model_performance.get('test_r2', 0.5)))
        
        # Adjust based on prediction magnitude
        price_change_pct = abs((prediction - current_price) / current_price)
        
        if price_change_pct > 0.2:  # Very large change
            confidence_multiplier = 0.6
        elif price_change_pct > 0.1:  # Large change
            confidence_multiplier = 0.8
        elif price_change_pct > 0.05:  # Moderate change
            confidence_multiplier = 0.9
        else:  # Small change
            confidence_multiplier = 1.0
        
        final_confidence = base_confidence * confidence_multiplier
        
        # Risk level assessment
        if final_confidence > 0.8:
            risk_level = "Low"
        elif final_confidence > 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return final_confidence, risk_level
    
    def _generate_llm_explanation(self, symbol: str, prediction: float, current_price: float, 
                                 technical_data: Dict, news_sentiment: Dict) -> str:
        """Generate human-readable explanation using OpenAI GPT"""
        
        if not self.openai_api_key:
            return self._generate_basic_explanation(symbol, prediction, current_price, technical_data)
        
        try:
            price_change = ((prediction - current_price) / current_price) * 100
            direction = "increase" if price_change > 0 else "decrease"
            
            prompt = f"""
            As a financial analyst, explain a stock prediction in simple terms:
            
            Stock: {symbol}
            Current Price: ${current_price:.2f}
            Predicted Price: ${prediction:.2f}
            Price Change: {price_change:.2f}%
            
            Technical Indicators:
            - RSI: {technical_data.get('RSI', 'N/A')}
            - Volume Trend: {technical_data.get('volume_trend', 'N/A')}
            - Trend Strength: {technical_data.get('trend_strength', 'N/A')}
            
            News Sentiment: {news_sentiment.get('sentiment_label', 'neutral')} ({news_sentiment.get('sentiment_score', 0):.2f})
            
            Provide a clear, concise explanation of why the model predicts this {direction}. 
            Include key factors and potential risks. Keep it under 150 words.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating LLM explanation: {str(e)}")
            return self._generate_basic_explanation(symbol, prediction, current_price, technical_data)
    
    def _generate_basic_explanation(self, symbol: str, prediction: float, current_price: float, technical_data: Dict) -> str:
        """Generate basic explanation without LLM"""
        
        price_change = ((prediction - current_price) / current_price) * 100
        direction = "increase" if price_change > 0 else "decrease"
        magnitude = "significant" if abs(price_change) > 5 else "moderate" if abs(price_change) > 2 else "slight"
        
        rsi = technical_data.get('RSI', 50)
        rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        
        return f"""
        The model predicts a {magnitude} {direction} of {abs(price_change):.1f}% for {symbol}. 
        This prediction is based on recent price patterns, technical indicators showing {rsi_signal} conditions (RSI: {rsi:.1f}), 
        and volume analysis. Key factors include trend momentum and market volatility patterns.
        """
    
    def predict_comprehensive(self, symbol: str, horizon: int = 1, user_level: str = "intermediate") -> PredictionResult:
        """Generate comprehensive prediction with detailed analysis"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Fetch latest data
        df = self._fetch_comprehensive_data(symbol, period="1y")
        current_price = float(df['Close'].iloc[-1])
        
        # Create features for prediction
        X, _, _ = self._make_advanced_features(df)
        
        # Make prediction using rolling approach for multi-day horizon
        last_features = X[-1].reshape(1, -1)
        predictions = []
        
        for step in range(horizon):
            pred = self.model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simplified approach)
            if step < horizon - 1:
                # This is a simplified feature update - in production, you'd want more sophisticated approach
                last_features = np.roll(last_features, -len(self.feature_names)//30)
                last_features[0, -len(self.feature_names)//30:] = pred
        
        final_prediction = float(predictions[-1])
        
        # Calculate confidence and risk
        confidence, risk_level = self._calculate_confidence_score(symbol, final_prediction, current_price)
        
        # Calculate prediction range based on confidence
        uncertainty = (1 - confidence) * 0.1 * current_price
        prediction_range = (final_prediction - uncertainty, final_prediction + uncertainty)
        
        # Get technical indicators for explanation
        technical_indicators = {
            'RSI': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None,
            'MACD': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns else None,
            'volume_trend': 'increasing' if df['Volume'].iloc[-5:].mean() > df['Volume'].iloc[-20:-5].mean() else 'decreasing',
            'trend_strength': 'strong' if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'weak',
            'volatility': float(df['Volatility'].iloc[-1]) if 'Volatility' in df.columns else None
        }
        
        # Get news sentiment
        news_sentiment = self._get_news_sentiment(symbol)
        
        # Market context
        market_context = {
            'current_price': current_price,
            'price_change_1d': float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100),
            'price_change_5d': float((df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100),
            'volume_vs_avg': float(df['Volume'].iloc[-1] / df['Volume_SMA'].iloc[-1]) if 'Volume_SMA' in df.columns else 1.0
        }
        
        # Generate explanation
        explanation = self._generate_llm_explanation(
            symbol, final_prediction, current_price, technical_indicators, news_sentiment
        )
        
        return PredictionResult(
            predicted_price=final_prediction,
            confidence_score=confidence,
            prediction_range=prediction_range,
            risk_level=risk_level,
            explanation=explanation,
            technical_indicators=technical_indicators,
            news_sentiment=news_sentiment,
            market_context=market_context,
            timeframe=horizon
        )
    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'model_params': self.model_params
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_performance = model_data['model_performance']
        self.model_params = model_data['model_params']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


# Legacy function wrapper for backward compatibility
def predict_stock_svr(df: pd.DataFrame, horizon: int = 1, lookback: int = 20) -> float:
    """Legacy function - use EnhancedSVMPredictor for new implementations"""
    
    def _make_features(df: pd.DataFrame, lookback: int = 20):
        X, y = [], []
        prices = df['Close'].values.flatten()
        
        for i in range(lookback, len(prices)):
            window = prices[i - lookback:i].flatten()
            if len(window) == lookback:
                X.append(window)
                y.append(prices[i])
        
        return np.array(X, dtype=float), np.array(y, dtype=float)
    
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < lookback + 2:
        raise ValueError("Not enough data to train SVR model.")
    
    X, y = _make_features(df, lookback=lookback)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])
    
    model.fit(X, y)
    
    # Roll forward predictions
    last_prices = df['Close'].values[-lookback:]
    window = [float(price) for price in last_prices]
    
    pred = None
    for step in range(horizon):
        current_window = window[-lookback:]
        x_next = np.array(current_window, dtype=float).reshape(1, -1)
        pred = float(model.predict(x_next)[0])
        window.append(pred)
        
        if len(window) > lookback * 2:
            window = window[-lookback-5:]
    
    return float(pred)