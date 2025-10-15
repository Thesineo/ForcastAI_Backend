"""
Prediction Service for AI Financial Advisor
Advanced ML predictions, technical analysis, and signal generation
Integrates forecasting, indicators, and signals into unified service
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import warnings
import joblib
import json
from functools import lru_cache
import time

# ML Libraries
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    
try:
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Import cache system
from .cache import CacheService, CacheType, get_cache_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionModel(Enum):
    """Available prediction models"""
    PROPHET = "prophet"
    SVR = "svr"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"
    LSTM = "lstm"

class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class TrendDirection(Enum):
    """Market trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

@dataclass
class TechnicalIndicators:
    """Technical indicators data structure"""
    symbol: str
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    stoch_k: float
    stoch_d: float
    atr: float
    adx: float
    cci: float
    williams_r: float
    momentum: float
    roc: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class PredictionResult:
    """Prediction result data structure"""
    symbol: str
    model_used: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence_score: float
    prediction_horizon: int
    technical_indicators: Optional[TechnicalIndicators]
    trend_direction: TrendDirection
    support_levels: List[float]
    resistance_levels: List[float]
    volatility_score: float
    model_accuracy: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal: SignalType
    confidence: float
    reasoning: List[str]
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    combined_score: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_level: str
    hold_period: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class PredictionService:
    """Advanced prediction and analysis service"""
    
    def __init__(self, cache_service: CacheService = None):
        self.cache = cache_service or get_cache_service()
        self.models = {}
        self.model_performance = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize prediction models"""
        logger.info("Initializing prediction models...")
        
        # Default model configurations
        self.model_configs = {
            PredictionModel.SVR: {
                'C': 100,
                'gamma': 'scale',
                'kernel': 'rbf',
                'epsilon': 0.01
            },
            PredictionModel.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            PredictionModel.XGBOOST: {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            } if HAS_XGB else None
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for technical indicators")
                return None
            
            df = df.copy()
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume'] if 'Volume' in df.columns else pd.Series([0] * len(df))
            
            # Your existing indicators (enhanced)
            indicators = self._calculate_core_indicators(df)
            
            # Additional advanced indicators
            indicators.update(self._calculate_advanced_indicators(df))
            
            # Get the latest values
            latest_indicators = TechnicalIndicators(
                symbol=df.get('symbol', 'UNKNOWN')[0] if 'symbol' in df.columns else 'UNKNOWN',
                rsi=float(indicators['RSI'].iloc[-1]) if not pd.isna(indicators['RSI'].iloc[-1]) else 50.0,
                macd=float(indicators['MACD'].iloc[-1]) if not pd.isna(indicators['MACD'].iloc[-1]) else 0.0,
                macd_signal=float(indicators['MACD_Signal'].iloc[-1]) if not pd.isna(indicators['MACD_Signal'].iloc[-1]) else 0.0,
                macd_histogram=float(indicators['MACD_Histogram'].iloc[-1]) if not pd.isna(indicators['MACD_Histogram'].iloc[-1]) else 0.0,
                sma_20=float(indicators['SMA_20'].iloc[-1]) if not pd.isna(indicators['SMA_20'].iloc[-1]) else close.iloc[-1],
                sma_50=float(indicators['SMA_50'].iloc[-1]) if not pd.isna(indicators['SMA_50'].iloc[-1]) else close.iloc[-1],
                ema_12=float(indicators['EMA_12'].iloc[-1]) if not pd.isna(indicators['EMA_12'].iloc[-1]) else close.iloc[-1],
                ema_26=float(indicators['EMA_26'].iloc[-1]) if not pd.isna(indicators['EMA_26'].iloc[-1]) else close.iloc[-1],
                bb_upper=float(indicators['BB_Upper'].iloc[-1]) if not pd.isna(indicators['BB_Upper'].iloc[-1]) else close.iloc[-1] * 1.02,
                bb_middle=float(indicators['BB_Middle'].iloc[-1]) if not pd.isna(indicators['BB_Middle'].iloc[-1]) else close.iloc[-1],
                bb_lower=float(indicators['BB_Lower'].iloc[-1]) if not pd.isna(indicators['BB_Lower'].iloc[-1]) else close.iloc[-1] * 0.98,
                bb_width=float(indicators['BB_Width'].iloc[-1]) if not pd.isna(indicators['BB_Width'].iloc[-1]) else 0.04,
                stoch_k=float(indicators['Stoch_K'].iloc[-1]) if not pd.isna(indicators['Stoch_K'].iloc[-1]) else 50.0,
                stoch_d=float(indicators['Stoch_D'].iloc[-1]) if not pd.isna(indicators['Stoch_D'].iloc[-1]) else 50.0,
                atr=float(indicators['ATR'].iloc[-1]) if not pd.isna(indicators['ATR'].iloc[-1]) else close.iloc[-1] * 0.02,
                adx=float(indicators['ADX'].iloc[-1]) if not pd.isna(indicators['ADX'].iloc[-1]) else 25.0,
                cci=float(indicators['CCI'].iloc[-1]) if not pd.isna(indicators['CCI'].iloc[-1]) else 0.0,
                williams_r=float(indicators['Williams_R'].iloc[-1]) if not pd.isna(indicators['Williams_R'].iloc[-1]) else -50.0,
                momentum=float(indicators['Momentum'].iloc[-1]) if not pd.isna(indicators['Momentum'].iloc[-1]) else 0.0,
                roc=float(indicators['ROC'].iloc[-1]) if not pd.isna(indicators['ROC'].iloc[-1]) else 0.0
            )
            
            return latest_indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    def _calculate_core_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate core technical indicators (your existing logic enhanced)"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        indicators = {}
        
        # RSI (your implementation enhanced)
        indicators['RSI'] = self._calculate_rsi(close, period=14)
        
        # MACD (your implementation enhanced)
        macd, signal, histogram = self._calculate_macd(close, fast=12, slow=26, signal_period=9)
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = signal
        indicators['MACD_Histogram'] = histogram
        
        # Moving Averages
        indicators['SMA_20'] = close.rolling(window=20).mean()
        indicators['SMA_50'] = close.rolling(window=50).mean()
        indicators['EMA_12'] = close.ewm(span=12).mean()
        indicators['EMA_26'] = close.ewm(span=26).mean()
        
        # Bollinger Bands
        sma_20 = indicators['SMA_20']
        std_20 = close.rolling(window=20).std()
        indicators['BB_Upper'] = sma_20 + (std_20 * 2)
        indicators['BB_Middle'] = sma_20
        indicators['BB_Lower'] = sma_20 - (std_20 * 2)
        indicators['BB_Width'] = indicators['BB_Upper'] - indicators['BB_Lower']
        
        return indicators
    
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        indicators = {}
        
        try:
            # Stochastic Oscillator
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            indicators['Stoch_K'] = stoch_k
            indicators['Stoch_D'] = stoch_k.rolling(window=3).mean()
            
            # Average True Range (ATR)
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['ATR'] = tr.rolling(window=14).mean()
            
            # Commodity Channel Index (CCI)
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (tp - sma_tp) / (0.015 * mad)
            
            # Williams %R
            indicators['Williams_R'] = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            # Momentum
            indicators['Momentum'] = close.diff(10)
            
            # Rate of Change (ROC)
            indicators['ROC'] = ((close / close.shift(12)) - 1) * 100
            
            # ADX (simplified version)
            indicators['ADX'] = self._calculate_adx(df)
            
        except Exception as e:
            logger.warning(f"Error calculating advanced indicators: {e}")
            # Fill with default values
            for key in ['Stoch_K', 'Stoch_D', 'ATR', 'CCI', 'Williams_R', 'Momentum', 'ROC', 'ADX']:
                if key not in indicators:
                    indicators[key] = pd.Series([50.0] * len(df), index=df.index)
        
        return indicators
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Enhanced RSI calculation (your implementation)"""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50.0] * len(series), index=series.index)
    
    def _calculate_macd(self, series: pd.Series, fast=12, slow=26, signal_period=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Enhanced MACD calculation (your implementation)"""
        try:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            zeros = pd.Series([0.0] * len(series), index=series.index)
            return zeros, zeros, zeros
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
            minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = ((dx.shift(1) * (period - 1)) + dx) / period
            adx_smooth = adx.ewm(alpha=1/period).mean()
            return adx_smooth
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return pd.Series([25.0] * len(df), index=df.index)
    
    def predict_with_prophet(self, df: pd.DataFrame, horizon_days: int = 1) -> Tuple[float, float]:
        """Enhanced Prophet prediction (your implementation)"""
        if not HAS_PROPHET:
            raise RuntimeError("Prophet not available")
        
        try:
            # Prepare data
            pdf = df[['Close']].reset_index()
            pdf.columns = ['ds', 'y']
            pdf['ds'] = pd.to_datetime(pdf['ds'])
            
            # Initialize and fit Prophet model
            m = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.8
            )
            
            m.fit(pdf)
            
            # Make future dataframe and predict
            future = m.make_future_dataframe(periods=horizon_days)
            forecast = m.predict(future)
            
            # Get prediction and confidence interval
            prediction = float(forecast['yhat'].iloc[-1])
            confidence = float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {e}")
            return float(df['Close'].iloc[-1]), 0.0
    
    def predict_with_svr(self, df: pd.DataFrame, horizon_days: int = 1) -> Tuple[float, float]:
        """Enhanced SVR prediction (your implementation enhanced)"""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn not available")
        
        try:
            # Prepare features
            features = self._prepare_ml_features(df)
            if len(features) < 30:  # Need sufficient data
                return float(df['Close'].iloc[-1]), 0.0
            
            # Prepare target (next day's close price)
            target = df['Close'].shift(-1).dropna()
            features = features.iloc[:-1]  # Align with target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVR model
            svr = SVR(**self.model_configs[PredictionModel.SVR])
            svr.fit(X_train_scaled, y_train)
            
            # Predict
            current_features = features.iloc[-1:].values
            current_features_scaled = scaler.transform(current_features)
            prediction = svr.predict(current_features_scaled)[0]
            
            # Calculate confidence based on model performance
            y_pred = svr.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            confidence = 1.0 / (1.0 + mse)  # Simple confidence measure
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in SVR prediction: {e}")
            return float(df['Close'].iloc[-1]), 0.0
    
    def predict_with_random_forest(self, df: pd.DataFrame, horizon_days: int = 1) -> Tuple[float, float]:
        """Random Forest prediction"""
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn not available")
        
        try:
            features = self._prepare_ml_features(df)
            if len(features) < 30:
                return float(df['Close'].iloc[-1]), 0.0
            
            target = df['Close'].shift(-1).dropna()
            features = features.iloc[:-1]
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, shuffle=False
            )
            
            rf = RandomForestRegressor(**self.model_configs[PredictionModel.RANDOM_FOREST])
            rf.fit(X_train, y_train)
            
            current_features = features.iloc[-1:].values
            prediction = rf.predict(current_features)[0]
            
            # Calculate confidence
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            confidence = max(0.0, r2)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return float(df['Close'].iloc[-1]), 0.0
    
    def predict_with_xgboost(self, df: pd.DataFrame, horizon_days: int = 1) -> Tuple[float, float]:
        """XGBoost prediction"""
        if not HAS_XGB:
            raise RuntimeError("XGBoost not available")
        
        try:
            features = self._prepare_ml_features(df)
            if len(features) < 30:
                return float(df['Close'].iloc[-1]), 0.0
            
            target = df['Close'].shift(-1).dropna()
            features = features.iloc[:-1]
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, shuffle=False
            )
            
            model = xgb.XGBRegressor(**self.model_configs[PredictionModel.XGBOOST])
            model.fit(X_train, y_train)
            
            current_features = features.iloc[-1:].values
            prediction = model.predict(current_features)[0]
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            confidence = max(0.0, r2)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {e}")
            return float(df['Close'].iloc[-1]), 0.0
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Price features
            features['close'] = df['Close']
            features['high'] = df['High']
            features['low'] = df['Low']
            features['volume'] = df['Volume'] if 'Volume' in df.columns else 0
            
            # Technical indicators
            indicators = self._calculate_core_indicators(df)
            for name, series in indicators.items():
                features[name.lower()] = series
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = df['Close'].shift(lag)
                features[f'volume_lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
                features[f'close_std_{window}'] = df['Close'].rolling(window).std()
                features[f'volume_mean_{window}'] = df['Volume'].rolling(window).mean() if 'Volume' in df.columns else 0
            
            # Price ratios
            features['high_low_ratio'] = df['High'] / df['Low']
            features['close_open_ratio'] = df['Close'] / df['Open'] if 'Open' in df.columns else 1
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return pd.DataFrame()
    
    def predict_stock_price(self, 
                          df: pd.DataFrame, 
                          symbol: str,
                          horizon_days: int = 1, 
                          model: Union[str, PredictionModel] = PredictionModel.ENSEMBLE) -> Optional[PredictionResult]:
        """Main prediction method (enhanced version of your forecast function)"""
        start_time = time.time()
        
        try:
            if isinstance(model, str):
                model = PredictionModel(model)
            
            current_price = float(df['Close'].iloc[-1])
            
            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(df)
            
            # Get prediction based on model
            if model == PredictionModel.ENSEMBLE:
                predicted_price, confidence = self._ensemble_predict(df, horizon_days)
                model_used = "ensemble"
            elif model == PredictionModel.PROPHET and HAS_PROPHET:
                predicted_price, confidence = self.predict_with_prophet(df, horizon_days)
                model_used = "prophet"
            elif model == PredictionModel.SVR:
                predicted_price, confidence = self.predict_with_svr(df, horizon_days)
                model_used = "svr"
            elif model == PredictionModel.RANDOM_FOREST:
                predicted_price, confidence = self.predict_with_random_forest(df, horizon_days)
                model_used = "random_forest"
            elif model == PredictionModel.XGBOOST and HAS_XGB:
                predicted_price, confidence = self.predict_with_xgboost(df, horizon_days)
                model_used = "xgboost"
            else:
                # Fallback to SVR (your default)
                predicted_price, confidence = self.predict_with_svr(df, horizon_days)
                model_used = "svr"
            
            # Calculate metrics
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Determine trend direction
            trend_direction = self._determine_trend(df, technical_indicators)
            
            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(df)
            
            # Calculate volatility score
            volatility_score = self._calculate_volatility_score(df)
            
            # Model accuracy (simplified)
            model_accuracy = min(0.95, max(0.5, confidence))
            
            result = PredictionResult(
                symbol=symbol.upper(),
                model_used=model_used,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                confidence_score=confidence,
                prediction_horizon=horizon_days,
                technical_indicators=technical_indicators,
                trend_direction=trend_direction,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volatility_score=volatility_score,
                model_accuracy=model_accuracy
            )
            
            # Cache the result
            cache_key = f"{symbol}_{model_used}_{horizon_days}d"
            self.cache.set(CacheType.PREDICTIONS, cache_key, asdict(result), ttl=1800)
            
            execution_time = time.time() - start_time
            logger.info(f"Predicted {symbol} price: ${predicted_price:.2f} ({price_change_percent:+.2f}%) in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting stock price for {symbol}: {e}")
            return None
    
    def _ensemble_predict(self, df: pd.DataFrame, horizon_days: int) -> Tuple[float, float]:
        """Ensemble prediction combining multiple models"""
        predictions = []
        confidences = []
        
        # Try different models
        models_to_try = [
            (PredictionModel.SVR, self.predict_with_svr),
            (PredictionModel.RANDOM_FOREST, self.predict_with_random_forest)
        ]
        
        if HAS_PROPHET:
            models_to_try.append((PredictionModel.PROPHET, self.predict_with_prophet))
        
        if HAS_XGB:
            models_to_try.append((PredictionModel.XGBOOST, self.predict_with_xgboost))
        
        for model_type, predict_func in models_to_try:
            try:
                pred, conf = predict_func(df, horizon_days)
                predictions.append(pred)
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")
        
        if not predictions:
            # Fallback to last price
            return float(df['Close'].iloc[-1]), 0.5
        
        # Weighted average based on confidence
        total_conf = sum(confidences)
        if total_conf > 0:
            weighted_pred = sum(p * c for p, c in zip(predictions, confidences)) / total_conf
            avg_conf = np.mean(confidences)
        else:
            weighted_pred = np.mean(predictions)
            avg_conf = 0.5
        
        return float(weighted_pred), float(avg_conf)
    
    def generate_trading_signal(self, 
                              df: pd.DataFrame, 
                              symbol: str,
                              prediction_result: Optional[PredictionResult] = None,
                              sentiment_score: float = 0.0,
                              fundamental_score: float = 0.0) -> Optional[TradingSignal]:
        """Enhanced signal generation (enhanced version of your generate_signal function)"""
        try:
            # Get prediction if not provided
            if prediction_result is None:
                prediction_result = self.predict_stock_price(df, symbol)
                if not prediction_result:
                    return None
            
            current_price = prediction_result.current_price
            predicted_price = prediction_result.predicted_price
            technical_indicators = prediction_result.technical_indicators
            
            # Technical analysis scoring
            technical_score = self._calculate_technical_score(technical_indicators, current_price)
            
            # Combined scoring
            combined_score = (
                technical_score * 0.4 + 
                sentiment_score * 0.3 + 
                fundamental_score * 0.2 + 
                (prediction_result.confidence_score * 0.1)
            )
            
            # Generate signal based on combined analysis
            signal, reasoning = self._determine_signal(
                current_price, predicted_price, technical_indicators, 
                technical_score, sentiment_score, combined_score
            )
            
            # Calculate target price and stop loss
            target_price, stop_loss = self._calculate_targets(
                current_price, predicted_price, technical_indicators, signal
            )
            
            # Determine risk level and hold period
            risk_level = self._assess_risk_level(prediction_result.volatility_score, combined_score)
            hold_period = self._determine_hold_period(signal, prediction_result.trend_direction)
            
            # Calculate confidence
            confidence = min(0.95, max(0.1, abs(combined_score)))
            
            trading_signal = TradingSignal(
                symbol=symbol.upper(),
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                combined_score=combined_score,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_level=risk_level,
                hold_period=hold_period
            )
            
            # Cache the signal
            cache_key = f"{symbol}_signal"
            self.cache.set(CacheType.PREDICTIONS, cache_key, asdict(trading_signal), ttl=900)  # 15 minutes
            
            logger.info(f"Generated {signal.value} signal for {symbol} (confidence: {confidence:.2f})")
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, indicators: TechnicalIndicators, current_price: float) -> float:
        """Calculate technical analysis score"""
        if not indicators:
            return 0.0
        
        score = 0.0
        
        try:
            # RSI scoring
            if indicators.rsi < 30:
                score += 0.15  # Oversold - bullish
            elif indicators.rsi > 70:
                score -= 0.15  # Overbought - bearish
            
            # MACD scoring
            if indicators.macd > indicators.macd_signal:
                score += 0.1  # Bullish crossover
            else:
                score -= 0.1  # Bearish crossover
            
            if indicators.macd_histogram > 0:
                score += 0.05  # Positive momentum
            else:
                score -= 0.05  # Negative momentum
            
            # Moving Average scoring
            if current_price > indicators.sma_20 > indicators.sma_50:
                score += 0.15  # Bullish trend
            elif current_price < indicators.sma_20 < indicators.sma_50:
                score -= 0.15  # Bearish trend
            
            # Bollinger Bands scoring
            bb_position = (current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
            if bb_position < 0.2:
                score += 0.1  # Near lower band - potential bounce
            elif bb_position > 0.8:
                score -= 0.1  # Near upper band - potential pullback
            
            # Stochastic scoring
            if indicators.stoch_k < 20 and indicators.stoch_d < 20:
                score += 0.1  # Oversold
            elif indicators.stoch_k > 80 and indicators.stoch_d > 80:
                score -= 0.1  # Overbought
            
            # ADX scoring (trend strength)
            if indicators.adx > 25:
                # Strong trend - amplify other signals
                score *= 1.2
            
            # Williams %R scoring
            if indicators.williams_r < -80:
                score += 0.05  # Oversold
            elif indicators.williams_r > -20:
                score -= 0.05  # Overbought
            
            return max(-1.0, min(1.0, score))  # Normalize to [-1, 1]
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.0
    
    def _determine_signal(self, 
                         current_price: float, 
                         predicted_price: float,
                         indicators: TechnicalIndicators,
                         technical_score: float,
                         sentiment_score: float,
                         combined_score: float) -> Tuple[SignalType, List[str]]:
        """Determine trading signal and reasoning"""
        reasoning = []
        
        try:
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Strong signals (enhanced version of your logic)
            if combined_score > 0.6 and price_change_pct > 3:
                reasoning.append(f"Strong bullish signals: {combined_score:.2f} combined score")
                reasoning.append(f"Price prediction: +{price_change_pct:.1f}%")
                return SignalType.STRONG_BUY, reasoning
            
            elif combined_score < -0.6 and price_change_pct < -3:
                reasoning.append(f"Strong bearish signals: {combined_score:.2f} combined score")
                reasoning.append(f"Price prediction: {price_change_pct:.1f}%")
                return SignalType.STRONG_SELL, reasoning
            
            # Medium signals
            elif combined_score > 0.3 and price_change_pct > 1:
                reasoning.append(f"Bullish technical indicators: {technical_score:.2f}")
                if sentiment_score > 0.3:
                    reasoning.append(f"Positive sentiment: {sentiment_score:.2f}")
                return SignalType.BUY, reasoning
            
            elif combined_score < -0.3 and price_change_pct < -1:
                reasoning.append(f"Bearish technical indicators: {technical_score:.2f}")
                if sentiment_score < -0.3:
                    reasoning.append(f"Negative sentiment: {sentiment_score:.2f}")
                return SignalType.SELL, reasoning
            
            # Hold conditions (your default logic enhanced)
            else:
                reasoning.append("Mixed signals or low conviction")
                if abs(price_change_pct) < 1:
                    reasoning.append("Limited price movement expected")
                if abs(technical_score) < 0.2:
                    reasoning.append("Neutral technical indicators")
                return SignalType.HOLD, reasoning
                
        except Exception as e:
            logger.error(f"Error determining signal: {e}")
            return SignalType.HOLD, ["Error in signal calculation"]
    
    def _calculate_targets(self, 
                         current_price: float,
                         predicted_price: float, 
                         indicators: TechnicalIndicators,
                         signal: SignalType) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss levels"""
        try:
            if signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                # Target based on resistance levels and prediction
                target_multiplier = 1.5 if signal == SignalType.STRONG_BUY else 1.2
                target_price = predicted_price * target_multiplier
                
                # Stop loss below support or 5% below current
                stop_loss = min(
                    current_price * 0.95,
                    indicators.bb_lower if indicators else current_price * 0.93
                )
                
            elif signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                # Target below predicted price
                target_multiplier = 0.5 if signal == SignalType.STRONG_SELL else 0.8
                target_price = predicted_price * target_multiplier
                
                # Stop loss above resistance or 5% above current
                stop_loss = max(
                    current_price * 1.05,
                    indicators.bb_upper if indicators else current_price * 1.07
                )
                
            else:  # HOLD
                target_price = None
                stop_loss = None
            
            return target_price, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating targets: {e}")
            return None, None
    
    def _assess_risk_level(self, volatility_score: float, combined_score: float) -> str:
        """Assess risk level for the trade"""
        try:
            # High volatility or low confidence = high risk
            if volatility_score > 0.3 or abs(combined_score) < 0.2:
                return "high"
            elif volatility_score > 0.15 or abs(combined_score) < 0.4:
                return "medium"
            else:
                return "low"
        except:
            return "medium"
    
    def _determine_hold_period(self, signal: SignalType, trend: TrendDirection) -> str:
        """Determine recommended hold period"""
        try:
            if signal in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                if trend == TrendDirection.BULLISH:
                    return "medium-term"  # 2-4 weeks
                else:
                    return "short-term"   # 1-2 weeks
            elif signal in [SignalType.BUY, SignalType.SELL]:
                return "short-term"       # 1-2 weeks
            else:
                return "hold"             # No specific timeframe
        except:
            return "short-term"
    
    def _determine_trend(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> TrendDirection:
        """Determine overall trend direction"""
        try:
            if not indicators:
                return TrendDirection.SIDEWAYS
            
            current_price = df['Close'].iloc[-1]
            
            # Multiple trend indicators
            trend_signals = 0
            
            # Moving average trend
            if current_price > indicators.sma_20 > indicators.sma_50:
                trend_signals += 2
            elif current_price < indicators.sma_20 < indicators.sma_50:
                trend_signals -= 2
            
            # MACD trend
            if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
                trend_signals += 1
            elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
                trend_signals -= 1
            
            # ADX trend strength
            if indicators.adx > 25:
                # Strong trend, amplify signal
                trend_signals *= 1.5
            
            # Determine trend
            if trend_signals >= 2:
                return TrendDirection.BULLISH
            elif trend_signals <= -2:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return TrendDirection.SIDEWAYS
    
    def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        try:
            if len(df) < lookback:
                current_price = df['Close'].iloc[-1]
                return [current_price * 0.95], [current_price * 1.05]
            
            recent_data = df.tail(lookback)
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            # Simple pivot point calculation
            for i in range(2, len(recent_data) - 2):
                # Resistance (local high)
                if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                    highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                    resistance_levels.append(highs.iloc[i])
                
                # Support (local low)
                if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                    lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                    support_levels.append(lows.iloc[i])
            
            # Sort and get most relevant levels
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels, reverse=True)[:3]
            
            # Fallback if no levels found
            if not resistance_levels:
                current_price = df['Close'].iloc[-1]
                resistance_levels = [current_price * 1.05, current_price * 1.10]
            
            if not support_levels:
                current_price = df['Close'].iloc[-1]
                support_levels = [current_price * 0.95, current_price * 0.90]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            current_price = df['Close'].iloc[-1]
            return [current_price * 0.95], [current_price * 1.05]
    
    def _calculate_volatility_score(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate volatility score"""
        try:
            if len(df) < period:
                return 0.2  # Default medium volatility
            
            # Calculate rolling standard deviation of returns
            returns = df['Close'].pct_change().dropna()
            volatility = returns.rolling(period).std().iloc[-1]
            
            # Normalize to 0-1 scale (0.5 = normal volatility for stocks)
            normalized_vol = min(1.0, volatility / 0.05)  # 5% daily volatility = 1.0 score
            
            return float(normalized_vol)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.2
    
    def analyze_stock_comprehensive(self, 
                                  df: pd.DataFrame, 
                                  symbol: str,
                                  sentiment_score: float = 0.0,
                                  fundamental_score: float = 0.0) -> Dict[str, Any]:
        """Comprehensive stock analysis combining predictions and signals"""
        start_time = time.time()
        
        try:
            # Get prediction
            prediction = self.predict_stock_price(df, symbol, horizon_days=1, model=PredictionModel.ENSEMBLE)
            if not prediction:
                return {'error': 'Prediction failed'}
            
            # Generate trading signal
            signal = self.generate_trading_signal(df, symbol, prediction, sentiment_score, fundamental_score)
            if not signal:
                return {'error': 'Signal generation failed'}
            
            # Additional analysis
            analysis = {
                'symbol': symbol.upper(),
                'prediction': asdict(prediction),
                'trading_signal': asdict(signal),
                'market_context': {
                    'trend_direction': prediction.trend_direction.value,
                    'volatility_level': 'high' if prediction.volatility_score > 0.3 else 'medium' if prediction.volatility_score > 0.15 else 'low',
                    'support_levels': prediction.support_levels,
                    'resistance_levels': prediction.resistance_levels
                },
                'risk_assessment': {
                    'overall_risk': signal.risk_level,
                    'volatility_score': prediction.volatility_score,
                    'confidence_level': signal.confidence
                },
                'recommendations': {
                    'action': signal.signal.value,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'hold_period': signal.hold_period,
                    'reasoning': signal.reasoning
                },
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Cache comprehensive analysis
            self.cache.set(CacheType.ANALYSIS, f"{symbol}_comprehensive", analysis, ttl=1800)
            
            logger.info(f"Comprehensive analysis for {symbol} completed in {analysis['execution_time']:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            'available_models': [model.value for model in PredictionModel],
            'model_status': {
                'prophet': HAS_PROPHET,
                'sklearn': HAS_SKLEARN,
                'xgboost': HAS_XGB
            },
            'performance_metrics': self.model_performance,
            'cache_stats': self.cache.get_stats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform prediction service health check"""
        try:
            start_time = time.time()
            
            # Create test data
            test_data = pd.DataFrame({
                'Open': [100] * 100,
                'High': [102] * 100,
                'Low': [98] * 100,
                'Close': [100 + i * 0.1 for i in range(100)],
                'Volume': [1000] * 100
            })
            
            # Test technical indicators
            indicators = self.calculate_technical_indicators(test_data)
            indicators_ok = indicators is not None
            
            # Test prediction
            prediction = self.predict_stock_price(test_data, "TEST", model=PredictionModel.SVR)
            prediction_ok = prediction is not None
            
            response_time = time.time() - start_time
            overall_healthy = indicators_ok and prediction_ok
            
            return {
                'service': 'prediction_service',
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'components': {
                    'technical_indicators': 'healthy' if indicators_ok else 'unhealthy',
                    'prediction_engine': 'healthy' if prediction_ok else 'unhealthy',
                    'cache_service': 'healthy' if self.cache.get_stats().get('status') == 'connected' else 'unhealthy'
                },
                'model_status': {
                    'prophet': HAS_PROPHET,
                    'sklearn': HAS_SKLEARN,
                    'xgboost': HAS_XGB
                },
                'response_time': round(response_time, 3),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'prediction_service',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

# Legacy compatibility functions (enhanced versions of your original functions)
def forecast(df: pd.DataFrame, horizon_days: int = 1, model: str = "svr") -> Tuple[float, str]:
    """Legacy forecast function for backward compatibility (enhanced)"""
    service = PredictionService()
    
    try:
        model_enum = PredictionModel(model)
    except ValueError:
        model_enum = PredictionModel.SVR
    
    prediction = service.predict_stock_price(df, "LEGACY", horizon_days, model_enum)
    if prediction:
        return prediction.predicted_price, prediction.model_used
    else:
        return float(df['Close'].iloc[-1]), "error"

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy add_indicators function (enhanced)"""
    service = PredictionService()
    indicators_dict = service._calculate_core_indicators(df)
    
    df_result = df.copy()
    for name, series in indicators_dict.items():
        df_result[name] = series
    
    return df_result.dropna()

def generate_signal(data, forecast_price, sentiment_score) -> str:
    """Legacy generate_signal function (enhanced)"""
    service = PredictionService()
    
    try:
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if df.empty:
            return "HOLD"
        
        # Create a mock prediction result
        current_price = df['Close'].iloc[-1] if 'Close' in df.columns else df.iloc[-1, 0]
        
        # Use the enhanced signal generation
        signal = service.generate_trading_signal(
            df, "LEGACY", None, sentiment_score, 0.0
        )
        
        if signal:
            return signal.signal.value
        else:
            return "HOLD"
            
    except Exception as e:
        logger.error(f"Error in legacy generate_signal: {e}")
        return "HOLD"

# Global service instance
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get global prediction service instance"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service

# Convenience functions
def predict_stock_price(df: pd.DataFrame, symbol: str, days: int = 1, model: str = "ensemble") -> Optional[PredictionResult]:
    """Convenience function for stock price prediction"""
    service = get_prediction_service()
    try:
        model_enum = PredictionModel(model)
    except ValueError:
        model_enum = PredictionModel.ENSEMBLE
    
    return service.predict_stock_price(df, symbol, days, model_enum)

def get_trading_signal(df: pd.DataFrame, symbol: str, sentiment: float = 0.0) -> Optional[TradingSignal]:
    """Convenience function for trading signal generation"""
    service = get_prediction_service()
    return service.generate_trading_signal(df, symbol, None, sentiment, 0.0)

def comprehensive_analysis(df: pd.DataFrame, symbol: str, sentiment: float = 0.0) -> Dict[str, Any]:
    """Convenience function for comprehensive analysis"""
    service = get_prediction_service()
    return service.analyze_stock_comprehensive(df, symbol, sentiment, 0.0)

if __name__ == "__main__":
    # Example usage and testing
    import yfinance as yf
    
    print("Testing Prediction Service...")
    
    # Get test data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    if not df.empty:
        service = PredictionService()
        
        # Test technical indicators
        indicators = service.calculate_technical_indicators(df)
        if indicators:
            print(f"Technical Indicators - RSI: {indicators.rsi:.2f}, MACD: {indicators.macd:.4f}")
        
        # Test prediction
        prediction = service.predict_stock_price(df, "AAPL", horizon_days=1)
        if prediction:
            print(f"Prediction: ${prediction.predicted_price:.2f} ({prediction.price_change_percent:+.2f}%)")
            print(f"Confidence: {prediction.confidence_score:.2f}, Model: {prediction.model_used}")
        
        # Test signal generation
        signal = service.generate_trading_signal(df, "AAPL", prediction)
        if signal:
            print(f"Trading Signal: {signal.signal.value} (Confidence: {signal.confidence:.2f})")
            print(f"Reasoning: {signal.reasoning}")
        
        # Test comprehensive analysis
        analysis = service.analyze_stock_comprehensive(df, "AAPL", sentiment_score=0.2)
        if 'error' not in analysis:
            print(f"Comprehensive Analysis completed in {analysis['execution_time']:.2f}s")
        
        # Health check
        health = service.health_check()
        print(f"Health Check: {health['status']}")
    else:
        print("No test data available")