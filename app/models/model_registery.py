import os
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import pandas as pd
from pathlib import Path

# Import your models
from .svm_models import EnhancedSVMPredictor
from .lstm_forecaster import LSTMForecaster
from .ensemble_predictor import EnsemblePredictor
from .risk_analyzer import RiskAnalyzer
from .sentiment_analyzer import SentimentAnalyzer

@dataclass
class ModelMetadata:
    """Metadata for registered models"""
    model_id: str
    model_type: str
    version: str
    symbol: str
    created_date: str
    last_updated: str
    performance_metrics: Dict
    hyperparameters: Dict
    training_data_hash: str
    file_path: str
    status: str  # 'active', 'deprecated', 'training', 'failed'
    tags: List[str]
    description: str

class ModelRegistry:
    """Centralized registry for managing ML models in production"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        # Registry database (JSON file for simplicity)
        self.registry_db_path = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
        
        # Model type mapping
        self.model_classes = {
            'svm': EnhancedSVMPredictor,
            'lstm': LSTMForecaster,
            'ensemble': EnsemblePredictor,
            'risk_analyzer': RiskAnalyzer,
            'sentiment_analyzer': SentimentAnalyzer
        }
        
        # Performance thresholds for model validation
        self.performance_thresholds = {
            'svm': {'min_r2': 0.3, 'max_mae': 0.1},
            'lstm': {'min_r2': 0.4, 'max_mae': 0.08},
            'ensemble': {'min_r2': 0.5, 'max_mae': 0.06}
        }
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.registry_db_path.exists():
            try:
                with open(self.registry_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_db_path, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def _generate_model_id(self, model_type: str, symbol: str, version: str = None) -> str:
        """Generate unique model ID"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_id = f"{model_type}_{symbol}_{version}"
        return base_id.lower().replace(' ', '_')
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of training data for versioning"""
        try:
            if isinstance(data, pd.DataFrame):
                data_str = data.to_string()
            elif isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return "unknown_hash"
    
    def register_model(self, model: Any, model_type: str, symbol: str, 
                      performance_metrics: Dict, hyperparameters: Dict = None,
                      training_data_hash: str = None, tags: List[str] = None,
                      description: str = None, version: str = None) -> str:
        """Register a trained model in the registry"""
        
        # Generate model ID
        model_id = self._generate_model_id(model_type, symbol, version)
        
        # Create model file path
        model_file_path = self.models_dir / f"{model_id}.pkl"
        
        # Validate model performance
        if not self._validate_model_performance(model_type, performance_metrics):
            status = "failed"
            print(f"Warning: Model {model_id} did not meet performance thresholds")
        else:
            status = "active"
        
        # Save model to disk
        try:
            if hasattr(model, 'save_model'):
                # Use model's built-in save method
                model.save_model(str(model_file_path.with_suffix('')))
            else:
                # Use joblib for generic objects
                joblib.dump(model, model_file_path)
            
            print(f"Model saved to {model_file_path}")
        except Exception as e:
            print(f"Error saving model {model_id}: {e}")
            status = "failed"
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
            symbol=symbol,
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters or {},
            training_data_hash=training_data_hash or "unknown",
            file_path=str(model_file_path),
            status=status,
            tags=tags or [],
            description=description or f"{model_type} model for {symbol}"
        )
        
        # Add to registry
        self.registry[model_id] = asdict(metadata)
        
        # If this is the best performing model of its type for this symbol, mark it as active
        self._update_active_model(symbol, model_type, model_id, performance_metrics)
        
        # Save registry
        self._save_registry()
        
        return model_id
    
    def _validate_model_performance(self, model_type: str, performance_metrics: Dict) -> bool:
        """Validate that model meets minimum performance requirements"""
        
        if model_type not in self.performance_thresholds:
            return True  # No thresholds defined, assume valid
        
        thresholds = self.performance_thresholds[model_type]
        
        # Check R² score
        if 'min_r2' in thresholds:
            r2_score = performance_metrics.get('test_r2', performance_metrics.get('val_r2', 0))
            if r2_score < thresholds['min_r2']:
                return False
        
        # Check MAE
        if 'max_mae' in thresholds:
            mae_score = performance_metrics.get('test_mae', performance_metrics.get('val_mae', float('inf')))
            if mae_score > thresholds['max_mae']:
                return False
        
        return True
    
    def _update_active_model(self, symbol: str, model_type: str, new_model_id: str, 
                            performance_metrics: Dict):
        """Update which model is marked as active for a symbol/type combination"""
        
        # Find current active model for this symbol/type
        current_active = None
        best_score = -float('inf')
        
        for model_id, metadata in self.registry.items():
            if (metadata['symbol'] == symbol and 
                metadata['model_type'] == model_type and 
                metadata['status'] == 'active'):
                
                # Use R² score as primary metric for comparison
                score = metadata['performance_metrics'].get('test_r2', 
                         metadata['performance_metrics'].get('val_r2', 0))
                
                if score > best_score:
                    best_score = score
                    current_active = model_id
        
        # Compare with new model
        new_score = performance_metrics.get('test_r2', performance_metrics.get('val_r2', 0))
        
        if new_score > best_score:
            # New model is better, deactivate old one
            if current_active:
                self.registry[current_active]['status'] = 'deprecated'
            
            # New model was already marked as active in register_model
        else:
            # New model is not better, mark as deprecated
            self.registry[new_model_id]['status'] = 'deprecated'
    
    def load_model(self, model_id: str) -> Any:
        """Load a model from the registry"""
        
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.registry[model_id]
        file_path = metadata['file_path']
        model_type = metadata['model_type']
        
        try:
            if model_type in self.model_classes:
                # Create new instance and load
                model_class = self.model_classes[model_type]
                model = model_class()
                
                if hasattr(model, 'load_model'):
                    # Use model's built-in load method
                    base_path = file_path.replace('.pkl', '')
                    model.load_model(base_path)
                else:
                    # Use joblib for generic objects
                    model = joblib.load(file_path)
            else:
                # Generic joblib load
                model = joblib.load(file_path)
            
            return model
            
        except Exception as e:
            raise Exception(f"Error loading model {model_id}: {e}")
    
    def get_active_model(self, symbol: str, model_type: str) -> Optional[Any]:
        """Get the currently active model for a symbol and type"""
        
        active_model_id = None
        
        for model_id, metadata in self.registry.items():
            if (metadata['symbol'] == symbol and 
                metadata['model_type'] == model_type and 
                metadata['status'] == 'active'):
                active_model_id = model_id
                break
        
        if active_model_id:
            return self.load_model(active_model_id)
        else:
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Get metadata for a specific model"""
        return self.registry.get(model_id)
    
    def list_models(self, symbol: str = None, model_type: str = None, 
                   status: str = None) -> List[Dict]:
        """List models with optional filtering"""
        
        filtered_models = []
        
        for model_id, metadata in self.registry.items():
            # Apply filters
            if symbol and metadata['symbol'] != symbol:
                continue
            if model_type and metadata['model_type'] != model_type:
                continue
            if status and metadata['status'] != status:
                continue
            
            filtered_models.append({
                'model_id': model_id,
                **metadata
            })
        
        # Sort by creation date (newest first)
        filtered_models.sort(key=lambda x: x['created_date'], reverse=True)
        
        return filtered_models
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """Compare performance metrics across multiple models"""
        
        comparison_data = {
            'models': {},
            'metrics_comparison': {},
            'best_model': None
        }
        
        metrics_to_compare = ['test_r2', 'val_r2', 'test_mae', 'val_mae']
        metric_values = {metric: [] for metric in metrics_to_compare}
        
        for model_id in model_ids:
            if model_id not in self.registry:
                continue
            
            metadata = self.registry[model_id]
            comparison_data['models'][model_id] = {
                'model_type': metadata['model_type'],
                'symbol': metadata['symbol'],
                'created_date': metadata['created_date'],
                'status': metadata['status'],
                'performance_metrics': metadata['performance_metrics']
            }
            
            # Collect metric values
            for metric in metrics_to_compare:
                value = metadata['performance_metrics'].get(metric)
                if value is not None:
                    metric_values[metric].append((model_id, value))
        
        # Find best model for each metric
        for metric, values in metric_values.items():
            if values:
                if 'mae' in metric.lower():
                    # For MAE, lower is better
                    best = min(values, key=lambda x: x[1])
                else:
                    # For R², higher is better
                    best = max(values, key=lambda x: x[1])
                
                comparison_data['metrics_comparison'][metric] = {
                    'best_model': best[0],
                    'best_value': best[1],
                    'all_values': {model_id: value for model_id, value in values}
                }
        
        # Determine overall best model (based on R² score)
        r2_metrics = [m for m in metric_values.keys() if 'r2' in m]
        if r2_metrics and metric_values[r2_metrics[0]]:
            best_overall = max(metric_values[r2_metrics[0]], key=lambda x: x[1])
            comparison_data['best_model'] = best_overall[0]
        
        return comparison_data
    
    def cleanup_old_models(self, keep_days: int = 30, keep_best: bool = True):
        """Remove old model files and registry entries"""
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        models_to_remove = []
        
        for model_id, metadata in self.registry.items():
            created_date = datetime.fromisoformat(metadata['created_date'])
            
            # Keep active models and recent models
            if metadata['status'] == 'active' or created_date > cutoff_date:
                continue
            
            # Keep best performing models if requested
            if keep_best:
                symbol = metadata['symbol']
                model_type = metadata['model_type']
                
                # Check if this is the best performing model for its type/symbol
                best_score = metadata['performance_metrics'].get('test_r2', 
                             metadata['performance_metrics'].get('val_r2', 0))
                
                is_best = True
                for other_id, other_metadata in self.registry.items():
                    if (other_id != model_id and 
                        other_metadata['symbol'] == symbol and 
                        other_metadata['model_type'] == model_type):
                        
                        other_score = other_metadata['performance_metrics'].get('test_r2',
                                     other_metadata['performance_metrics'].get('val_r2', 0))
                        
                        if other_score > best_score:
                            is_best = False
                            break
                
                if is_best:
                    continue
            
            models_to_remove.append(model_id)
        
        # Remove models
        removed_count = 0
        for model_id in models_to_remove:
            try:
                # Remove file
                file_path = Path(self.registry[model_id]['file_path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from registry
                del self.registry[model_id]
                removed_count += 1
                
            except Exception as e:
                print(f"Error removing model {model_id}: {e}")
        
        # Save updated registry
        if removed_count > 0:
            self._save_registry()
            print(f"Removed {removed_count} old models")
        
        return removed_count
    
    def get_registry_stats(self) -> Dict:
        """Get statistics about the model registry"""
        
        stats = {
            'total_models': len(self.registry),
            'by_status': {},
            'by_type': {},
            'by_symbol': {},
            'storage_size_mb': 0
        }
        
        for model_id, metadata in self.registry.items():
            # Count by status
            status = metadata['status']
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by type
            model_type = metadata['model_type']
            stats['by_type'][model_type] = stats['by_type'].get(model_type, 0) + 1
            
            # Count by symbol
            symbol = metadata['symbol']
            stats['by_symbol'][symbol] = stats['by_symbol'].get(symbol, 0) + 1
            
            # Calculate storage size
            try:
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    stats['storage_size_mb'] += file_path.stat().st_size / (1024 * 1024)
            except Exception:
                pass
        
        stats['storage_size_mb'] = round(stats['storage_size_mb'], 2)
        
        return stats


# Convenience functions for common operations
def register_trained_model(model, model_type: str, symbol: str, registry_path: str = None) -> str:
    """Quick function to register a trained model"""
    registry = ModelRegistry(registry_path or "./model_registry")
    
    # Extract performance metrics from model
    if hasattr(model, 'model_performance'):
        performance_metrics = model.model_performance
    elif hasattr(model, 'ensemble_performance'):
        performance_metrics = model.ensemble_performance
    else:
        performance_metrics = {'note': 'No performance metrics available'}
    
    return registry.register_model(
        model=model,
        model_type=model_type,
        symbol=symbol,
        performance_metrics=performance_metrics
    )

def load_best_model(symbol: str, model_type: str, registry_path: str = None):
    """Quick function to load the best model for a symbol"""
    registry = ModelRegistry(registry_path or "./model_registry")
    return registry.get_active_model(symbol, model_type)