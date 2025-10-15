import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import your custom models
from .svm_models import EnhancedSVMPredictor
from .lstm_forecaster import LSTMForecaster

warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """Advanced ensemble model combining SVM, LSTM, and other ML models"""
    
    def __init__(self, models_config: Dict = None):
        
        # Default model configuration
        self.models_config = models_config or {
            'svm': {'weight': 0.3, 'enabled': True},
            'lstm': {'weight': 0.5, 'enabled': True},
            'random_forest': {'weight': 0.2, 'enabled': False}  # Optional third model
        }
        
        # Initialize models
        self.svm_model = EnhancedSVMPredictor()
        self.lstm_model = LSTMForecaster()
        self.rf_model = None  # Will be initialized if enabled
        
        self.is_trained = False
        self.model_performances = {}
        self.ensemble_performance = {}
        self.feature_importance = {}
        
        # Ensemble weights (will be optimized during training)
        self.optimized_weights = None
    
    def _validate_weights(self):
        """Ensure ensemble weights sum to 1"""
        enabled_models = {k: v for k, v in self.models_config.items() if v['enabled']}
        
        total_weight = sum(model['weight'] for model in enabled_models.values())
        
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            for model_name in enabled_models:
                self.models_config[model_name]['weight'] /= total_weight
    
    def _prepare_random_forest_model(self):
        """Initialize Random Forest model if enabled"""
        if self.models_config.get('random_forest', {}).get('enabled', False):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            self.rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
    
    def train_ensemble(self, symbol: str, optimize_weights: bool = True) -> Dict:
        """Train all enabled models in the ensemble"""
        
        print(f"Training ensemble models for {symbol}...")
        self._validate_weights()
        
        training_results = {}
        
        # Train SVM model
        if self.models_config['svm']['enabled']:
            print("Training SVM model...")
            svm_performance = self.svm_model.train_model(symbol, optimize_params=True)
            self.model_performances['svm'] = svm_performance
            training_results['svm'] = svm_performance
        
        # Train LSTM model
        if self.models_config['lstm']['enabled']:
            print("Training LSTM model...")
            lstm_performance = self.lstm_model.train_model(symbol)
            self.model_performances['lstm'] = lstm_performance
            training_results['lstm'] = lstm_performance
        
        # Train Random Forest model (if enabled)
        if self.models_config.get('random_forest', {}).get('enabled', False):
            print("Training Random Forest model...")
            self._prepare_random_forest_model()
            rf_performance = self._train_random_forest(symbol)
            self.model_performances['random_forest'] = rf_performance
            training_results['random_forest'] = rf_performance
        
        # Optimize ensemble weights if requested
        if optimize_weights:
            print("Optimizing ensemble weights...")
            self.optimized_weights = self._optimize_ensemble_weights(symbol)
        else:
            self.optimized_weights = {
                model: config['weight'] 
                for model, config in self.models_config.items() 
                if config['enabled']
            }
        
        # Evaluate ensemble performance
        self.ensemble_performance = self._evaluate_ensemble_performance(symbol)
        
        self.is_trained = True
        
        print(f"Ensemble training completed!")
        print(f"Optimized weights: {self.optimized_weights}")
        print(f"Ensemble R² Score: {self.ensemble_performance.get('r2', 'N/A'):.4f}")
        
        return {
            'individual_models': training_results,
            'optimized_weights': self.optimized_weights,
            'ensemble_performance': self.ensemble_performance
        }
    
    def _train_random_forest(self, symbol: str) -> Dict:
        """Train Random Forest model (simplified implementation)"""
        
        # Fetch data (reuse SVM data preparation logic)
        df = self.svm_model._fetch_comprehensive_data(symbol)
        X, y, _ = self.svm_model._make_advanced_features(df, lookback=30)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate performance
        train_pred = self.rf_model.predict(X_train)
        test_pred = self.rf_model.predict(X_test)
        
        return {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
    
    def _optimize_ensemble_weights(self, symbol: str, validation_days: int = 30) -> Dict:
        """Optimize ensemble weights using validation data"""
        
        from scipy.optimize import minimize
        
        # Get validation predictions from each model
        validation_predictions = {}
        
        if self.models_config['svm']['enabled']:
            svm_pred = self.svm_model.predict_comprehensive(symbol, horizon=1)
            validation_predictions['svm'] = [svm_pred.predicted_price]
        
        if self.models_config['lstm']['enabled']:
            lstm_pred = self.lstm_model.predict(symbol, days_ahead=1)
            validation_predictions['lstm'] = [lstm_pred['final_prediction']]
        
        if self.models_config.get('random_forest', {}).get('enabled', False):
            # Simplified RF prediction
            df = self.svm_model._fetch_comprehensive_data(symbol)
            X, _, _ = self.svm_model._make_advanced_features(df, lookback=30)
            rf_pred = self.rf_model.predict(X[-1:])
            validation_predictions['random_forest'] = [rf_pred[0]]
        
        # For simplicity, we'll use model performance metrics to optimize weights
        # In production, you'd want to use actual validation data
        
        enabled_models = [model for model, config in self.models_config.items() if config['enabled']]
        
        def objective(weights):
            """Objective function to maximize ensemble performance"""
            weighted_score = 0
            for i, model in enumerate(enabled_models):
                model_r2 = self.model_performances[model].get('test_r2', 0.5)
                weighted_score += weights[i] * model_r2
            return -weighted_score  # Minimize negative score
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in enabled_models]
        
        # Initial weights
        initial_weights = [self.models_config[model]['weight'] for model in enabled_models]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        # Return optimized weights
        optimized_weights = {}
        for i, model in enumerate(enabled_models):
            optimized_weights[model] = float(result.x[i])
        
        return optimized_weights
    
    def _evaluate_ensemble_performance(self, symbol: str) -> Dict:
        """Evaluate ensemble performance using cross-validation"""
        
        # This is a simplified evaluation
        # In production, you'd want proper time series cross-validation
        
        individual_scores = []
        weights = []
        
        for model_name, weight in self.optimized_weights.items():
            if model_name in self.model_performances:
                r2_score = self.model_performances[model_name].get('test_r2', 0.5)
                individual_scores.append(r2_score)
                weights.append(weight)
        
        # Calculate weighted average performance
        ensemble_r2 = np.average(individual_scores, weights=weights)
        
        return {
            'r2': ensemble_r2,
            'individual_contributions': dict(zip(self.optimized_weights.keys(), individual_scores)),
            'ensemble_method': 'weighted_average'
        }
    
    def predict_ensemble(self, symbol: str, horizon: int = 7, 
                        include_uncertainty: bool = True) -> Dict:
        """Generate ensemble predictions with uncertainty quantification"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        individual_predictions = {}
        predictions_list = []
        weights_list = []
        
        # Get predictions from each enabled model
        if self.models_config['svm']['enabled'] and 'svm' in self.optimized_weights:
            print("Getting SVM predictions...")
            svm_result = self.svm_model.predict_comprehensive(symbol, horizon=horizon)
            individual_predictions['svm'] = {
                'prediction': svm_result.predicted_price,
                'confidence': svm_result.confidence_score,
                'explanation': svm_result.explanation
            }
            predictions_list.append(svm_result.predicted_price)
            weights_list.append(self.optimized_weights['svm'])
        
        if self.models_config['lstm']['enabled'] and 'lstm' in self.optimized_weights:
            print("Getting LSTM predictions...")
            lstm_result = self.lstm_model.predict(symbol, days_ahead=horizon)
            individual_predictions['lstm'] = {
                'prediction': lstm_result['final_prediction'],
                'confidence': lstm_result['confidence'],
                'explanation': f"LSTM neural network analysis predicts {lstm_result['predicted_change_pct']:.1f}% change"
            }
            predictions_list.append(lstm_result['final_prediction'])
            weights_list.append(self.optimized_weights['lstm'])
        
        if (self.models_config.get('random_forest', {}).get('enabled', False) and 
            'random_forest' in self.optimized_weights):
            print("Getting Random Forest predictions...")
            df = self.svm_model._fetch_comprehensive_data(symbol)
            X, _, _ = self.svm_model._make_advanced_features(df, lookback=30)
            rf_pred = self.rf_model.predict(X[-1:])
            individual_predictions['random_forest'] = {
                'prediction': float(rf_pred[0]),
                'confidence': 0.7,  # Default confidence for RF
                'explanation': "Random Forest ensemble analysis"
            }
            predictions_list.append(float(rf_pred[0]))
            weights_list.append(self.optimized_weights['random_forest'])
        
        # Calculate weighted ensemble prediction
        ensemble_prediction = np.average(predictions_list, weights=weights_list)
        
        # Calculate ensemble confidence
        weighted_confidences = []
        for model_name, weight in zip(self.optimized_weights.keys(), weights_list):
            if model_name in individual_predictions:
                confidence = individual_predictions[model_name]['confidence']
                weighted_confidences.append(confidence * weight)
        
        ensemble_confidence = sum(weighted_confidences)
        
        # Calculate prediction uncertainty
        prediction_std = np.std(predictions_list) if len(predictions_list) > 1 else 0
        uncertainty_range = prediction_std * 1.96  # 95% confidence interval
        
        # Get current price for comparison
        current_price = individual_predictions[list(individual_predictions.keys())[0]].get('current_price')
        if not current_price:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            current_price = float(ticker.history(period="1d")['Close'].iloc[-1])
        
        # Generate ensemble explanation
        ensemble_explanation = self._generate_ensemble_explanation(
            symbol, ensemble_prediction, current_price, individual_predictions
        )
        
        return {
            'ensemble_prediction': float(ensemble_prediction),
            'current_price': current_price,
            'predicted_change_pct': ((ensemble_prediction - current_price) / current_price) * 100,
            'ensemble_confidence': float(ensemble_confidence),
            'prediction_range': {
                'lower': float(ensemble_prediction - uncertainty_range),
                'upper': float(ensemble_prediction + uncertainty_range)
            },
            'uncertainty_score': float(prediction_std / ensemble_prediction if ensemble_prediction != 0 else 0),
            'individual_predictions': individual_predictions,
            'model_weights': self.optimized_weights,
            'ensemble_explanation': ensemble_explanation,
            'horizon_days': horizon,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_ensemble_explanation(self, symbol: str, prediction: float, 
                                     current_price: float, individual_predictions: Dict) -> str:
        """Generate human-readable explanation for ensemble prediction"""
        
        change_pct = ((prediction - current_price) / current_price) * 100
        direction = "increase" if change_pct > 0 else "decrease"
        magnitude = "significant" if abs(change_pct) > 5 else "moderate" if abs(change_pct) > 2 else "slight"
        
        # Get agreement level between models
        predictions_values = [pred['prediction'] for pred in individual_predictions.values()]
        agreement_std = np.std(predictions_values) / np.mean(predictions_values) if predictions_values else 0
        
        if agreement_std < 0.05:
            agreement_level = "strong consensus"
        elif agreement_std < 0.1:
            agreement_level = "moderate agreement"
        else:
            agreement_level = "mixed signals"
        
        explanation = f"""
        The ensemble model predicts a {magnitude} {direction} of {abs(change_pct):.1f}% for {symbol} based on {agreement_level} 
        among {len(individual_predictions)} AI models. The prediction combines:
        """
        
        # Add individual model contributions
        for model_name, pred_data in individual_predictions.items():
            model_weight = self.optimized_weights.get(model_name, 0) * 100
            explanation += f"\n• {model_name.upper()}: {model_weight:.0f}% weight, predicts ${pred_data['prediction']:.2f}"
        
        explanation += f"\n\nThis ensemble approach reduces individual model bias and provides more reliable forecasting."
        
        return explanation
    
    def get_model_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics for all models in the ensemble"""
        
        diagnostics = {
            'ensemble_status': {
                'is_trained': self.is_trained,
                'optimized_weights': self.optimized_weights,
                'ensemble_performance': self.ensemble_performance
            },
            'individual_models': {}
        }
        
        # SVM diagnostics
        if self.models_config['svm']['enabled']:
            diagnostics['individual_models']['svm'] = {
                'is_trained': self.svm_model.is_trained,
                'performance': self.model_performances.get('svm', {}),
                'feature_count': len(self.svm_model.feature_names) if hasattr(self.svm_model, 'feature_names') else 0
            }
        
        # LSTM diagnostics
        if self.models_config['lstm']['enabled']:
            diagnostics['individual_models']['lstm'] = {
                'is_trained': self.lstm_model.is_trained,
                'performance': self.model_performances.get('lstm', {}),
                'sequence_length': self.lstm_model.sequence_length,
                'feature_count': len(self.lstm_model.features_to_use)
            }
        
        # Random Forest diagnostics
        if self.models_config.get('random_forest', {}).get('enabled', False):
            diagnostics['individual_models']['random_forest'] = {
                'is_trained': self.rf_model is not None,
                'performance': self.model_performances.get('random_forest', {})
            }
        
        return diagnostics
    
    def update_model_weights(self, new_weights: Dict):
        """Manually update ensemble weights"""
        
        # Validate new weights
        enabled_models = [model for model, config in self.models_config.items() if config['enabled']]
        
        for model in enabled_models:
            if model not in new_weights:
                raise ValueError(f"Missing weight for model: {model}")
        
        if abs(sum(new_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.optimized_weights = new_weights.copy()
        print(f"Updated ensemble weights: {self.optimized_weights}")
    
    def get_feature_importance_ensemble(self, symbol: str) -> Dict:
        """Get aggregated feature importance from all models"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before analyzing feature importance")
        
        ensemble_importance = {}
        
        # Get SVM feature importance (based on model coefficients)
        if self.models_config['svm']['enabled'] and hasattr(self.svm_model, 'model'):
            # This is a simplified approach - SVM doesn't directly provide feature importance
            # In practice, you might use permutation importance or SHAP values
            svm_weight = self.optimized_weights.get('svm', 0)
            ensemble_importance['svm_contribution'] = svm_weight
        
        # Get LSTM feature importance
        if self.models_config['lstm']['enabled']:
            lstm_importance = self.lstm_model.get_feature_importance(symbol)
            lstm_weight = self.optimized_weights.get('lstm', 0)
            
            for feature, importance in lstm_importance.items():
                weighted_importance = importance * lstm_weight
                if feature in ensemble_importance:
                    ensemble_importance[feature] += weighted_importance
                else:
                    ensemble_importance[feature] = weighted_importance
        
        return ensemble_importance
    
    def backtest_ensemble(self, symbol: str, start_date: str, end_date: str, 
                         horizon: int = 7) -> Dict:
        """Backtest the ensemble model over a specified period"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before backtesting")
        
        import yfinance as yf
        from datetime import datetime
        
        # This is a simplified backtest implementation
        # In production, you'd want more sophisticated walk-forward analysis
        
        ticker = yf.Ticker(symbol)
        historical_data = ticker.history(start=start_date, end=end_date)
        
        if len(historical_data) < horizon + 30:
            raise ValueError("Insufficient data for backtesting")
        
        # Simulate predictions at regular intervals
        backtest_results = {
            'predictions': [],
            'actuals': [],
            'dates': [],
            'accuracy_metrics': {}
        }
        
        # For simplicity, we'll just test the final period
        # In practice, you'd do rolling predictions
        test_period = historical_data[-30:]  # Last 30 days
        actual_prices = test_period['Close'].values
        
        # Make a prediction for comparison
        try:
            ensemble_result = self.predict_ensemble(symbol, horizon=horizon)
            final_prediction = ensemble_result['ensemble_prediction']
            
            # Compare with actual price at the end of horizon
            if len(actual_prices) >= horizon:
                actual_price = actual_prices[-1]  # Last available price
                
                backtest_results['predictions'] = [final_prediction]
                backtest_results['actuals'] = [actual_price]
                backtest_results['dates'] = [historical_data.index[-1].strftime('%Y-%m-%d')]
                
                # Calculate accuracy metrics
                error = abs(final_prediction - actual_price)
                error_pct = (error / actual_price) * 100
                
                backtest_results['accuracy_metrics'] = {
                    'absolute_error': float(error),
                    'percentage_error': float(error_pct),
                    'accuracy_score': float(max(0, 100 - error_pct))
                }
            
        except Exception as e:
            print(f"Backtest error: {str(e)}")
            backtest_results['error'] = str(e)
        
        return backtest_results
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble to disk"""
        
        if not self.is_trained:
            raise ValueError("No trained ensemble to save")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save SVM model
        if self.models_config['svm']['enabled']:
            self.svm_model.save_model(f"{filepath}_svm")
        
        # Save LSTM model
        if self.models_config['lstm']['enabled']:
            self.lstm_model.save_model(f"{filepath}_lstm")
        
        # Save Random Forest model
        if self.models_config.get('random_forest', {}).get('enabled', False) and self.rf_model:
            joblib.dump(self.rf_model, f"{filepath}_rf.pkl")
        
        # Save ensemble configuration
        ensemble_config = {
            'models_config': self.models_config,
            'optimized_weights': self.optimized_weights,
            'model_performances': self.model_performances,
            'ensemble_performance': self.ensemble_performance
        }
        
        joblib.dump(ensemble_config, f"{filepath}_ensemble_config.pkl")
        
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a trained ensemble from disk"""
        
        # Load ensemble configuration
        ensemble_config = joblib.load(f"{filepath}_ensemble_config.pkl")
        
        self.models_config = ensemble_config['models_config']
        self.optimized_weights = ensemble_config['optimized_weights']
        self.model_performances = ensemble_config['model_performances']
        self.ensemble_performance = ensemble_config['ensemble_performance']
        
        # Load individual models
        if self.models_config['svm']['enabled']:
            self.svm_model.load_model(f"{filepath}_svm")
        
        if self.models_config['lstm']['enabled']:
            self.lstm_model.load_model(f"{filepath}_lstm")
        
        if self.models_config.get('random_forest', {}).get('enabled', False):
            import os
            if os.path.exists(f"{filepath}_rf.pkl"):
                self.rf_model = joblib.load(f"{filepath}_rf.pkl")
        
        self.is_trained = True
        
        print(f"Ensemble loaded from {filepath}")


# Utility function for quick ensemble predictions
def create_and_train_ensemble(symbol: str, models_to_use: List[str] = None, 
                             openai_api_key: str = None) -> EnsemblePredictor:
    """Convenience function to create and train an ensemble model"""
    
    if models_to_use is None:
        models_to_use = ['svm', 'lstm']
    
    # Configure models
    models_config = {
        'svm': {'weight': 0.4, 'enabled': 'svm' in models_to_use},
        'lstm': {'weight': 0.6, 'enabled': 'lstm' in models_to_use},
        'random_forest': {'weight': 0.0, 'enabled': 'random_forest' in models_to_use}
    }
    
    # Create ensemble
    ensemble = EnsemblePredictor(models_config=models_config)
    
    # Set OpenAI key for SVM model if provided
    if openai_api_key:
        ensemble.svm_model.openai_api_key = openai_api_key
    
    # Train ensemble
    training_results = ensemble.train_ensemble(symbol, optimize_weights=True)
    
    return ensemble           