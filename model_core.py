# -*- coding: utf-8 -*-
"""
Model Core Module
Define model structure and algorithm implementations
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class ModelCore:
    """
    Machine Learning Model Core Class
    Supports multiple regression algorithms and hyperparameter configuration
    """
    
    def __init__(self, model_type='gradient_boosting', **kwargs):
        """
        Initialize model
        
        Parameters:
            model_type (str): Model type
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        self.model = None
        self.hyperparameters = kwargs
        self.is_fitted = False
        
        # 创建模型实例
        self._create_model()
    
    def _create_model(self):
        """
        Create corresponding model instance based on model type
        
        Returns:
            None
        """
        model_configs = {
            'gradient_boosting': {
                'class': GradientBoostingRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'random_forest': {
                'class': RandomForestRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'svr': {
                'class': SVR,
                'default_params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            },
            'linear_regression': {
                'class': LinearRegression,
                'default_params': {}
            },
            'ridge': {
                'class': Ridge,
                'default_params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            },
            'lasso': {
                'class': Lasso,
                'default_params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            },
            'elastic_net': {
                'class': ElasticNet,
                'default_params': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5,
                    'random_state': 42
                }
            },
            'neural_network': {
                'class': MLPRegressor,
                'default_params': {
                    'hidden_layer_sizes': (100, 50),
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'xgboost': {
                'class': xgb.XGBRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        config = model_configs[self.model_type]
        
        # Merge default parameters with user parameters
        params = {**config['default_params'], **self.hyperparameters}
        
        # Create model instance
        self.model = config['class'](**params)
        
        print(f"Created {self.model_type} model with parameters: {params}")
    
    def fit(self, X, y):
        """
        Train model
        
        Parameters:
            X: Training features
            y: Training targets
            
        Returns:
            self: Return self instance
        """
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            print(f"{self.model_type} model training completed")
            return self
        except Exception as e:
            print(f"Model training failed: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
            X: Input features
            
        Returns:
            np.array: Prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet, please call fit method first")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y, metric='mse'):
        """
        Evaluate model performance
        
        Parameters:
            X: Test features
            y: Test targets
            metric (str): Evaluation metric ('mse', 'rmse', 'mae', 'r2')
            
        Returns:
            float: Evaluation result
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet, please call fit method first")
        
        y_pred = self.predict(X)
        
        if metric == 'mse':
            return mean_squared_error(y, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y, y_pred)
        elif metric == 'r2':
            return r2_score(y, y_pred)
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
    
    def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Cross validation
        
        Parameters:
            X: Feature data
            y: Target data
            cv (int): Number of cross-validation folds
            scoring (str): Scoring method
            
        Returns:
            dict: Cross validation results
        """
        if not self.is_fitted:
            # Fit model before cross validation
            self.fit(X, y)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': -scores.mean(),  # Convert to positive value
            'std_score': scores.std(),
            'scores': -scores  # Convert to positive value
        }
        
        print(f"Cross validation results (CV={cv}):")
        print(f"Mean score: {results['mean_score']:.4f} (±{results['std_score']:.4f})")
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance (if model supports it)
        
        Returns:
            np.array: Feature importance array
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        # Models that support feature importance
        importance_models = [
            'gradient_boosting', 'random_forest', 'xgboost'
        ]
        
        if self.model_type in importance_models:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
        
        raise ValueError(f"{self.model_type} model does not support feature importance extraction")
    
    def get_model_info(self):
        """
        Get model information
        
        Returns:
            dict: Model information
        """
        info = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'hyperparameters': self.hyperparameters
        }
        
        if self.is_fitted:
            # Add training information
            info['n_features_in_'] = getattr(self.model, 'n_features_in_', None)
            info['n_outputs_'] = getattr(self.model, 'n_outputs_', 1)
        
        return info
    
    def save_model(self, filepath):
        """
        Save model
        
        Parameters:
            filepath (str): Save path
        """
        import pickle
        
        if not self.is_fitted:
            raise ValueError("Model not trained yet, cannot save")
        
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'hyperparameters': self.hyperparameters,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load model
        
        Parameters:
            filepath (str): Model file path
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.model = model_data['model']
        self.hyperparameters = model_data['hyperparameters']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")
    
    def __str__(self):
        return f"{self.model_type}模型 (已训练: {self.is_fitted})"
    
    def __repr__(self):
        return self.__str__()


class ModelSelector:
    """
    Model Selector Class
    Used to compare multiple models and select the best model
    """
    
    def __init__(self, models_to_try=None):
        """
        Initialize model selector
        
        Parameters:
            models_to_try (list): List of model types to try
        """
        if models_to_try is None:
            self.models_to_try = [
                'gradient_boosting', 'random_forest', 'xgboost', 
                'ridge', 'lasso', 'svr'
            ]
        else:
            self.models_to_try = models_to_try
        
        self.results = {}
        self.best_model = None
        self.best_model_type = None
    
    def compare_models(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Compare performance of multiple models
        
        Parameters:
            X: Feature data
            y: Target data
            cv (int): Number of cross-validation folds
            scoring (str): Scoring method
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        print("Starting model comparison...")
        
        for model_type in self.models_to_try:
            try:
                print(f"Testing {model_type} model...")
                
                # Create model instance
                model = ModelCore(model_type)
                
                # Cross validation
                cv_results = model.cross_validate(X, y, cv=cv, scoring=scoring)
                
                self.results[model_type] = cv_results
                
            except Exception as e:
                print(f"Model {model_type} test failed: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df = results_df.sort_values('mean_score')
        
        # Select best model
        if len(results_df) > 0:
            self.best_model_type = results_df.index[0]
            self.best_model = ModelCore(self.best_model_type)
            self.best_model.fit(X, y)
        
        print("\nModel comparison results:")
        print(results_df)
        
        return results_df
    
    def get_best_model(self):
        """
        Get best model
        
        Returns:
            ModelCore: Best model instance
        """
        return self.best_model