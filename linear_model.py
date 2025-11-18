# -*- coding: utf-8 -*-
"""
Linear Regression Model Module
Contains linear regression implementation for medical insurance cost prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionModel:
    """
    Linear Regression Model Class
    Supports linear regression, Ridge, Lasso, and Elastic Net regression
    """
    
    def __init__(self, model_type='linear', **kwargs):
        """
        Initialize the model
        
        Parameters:
            model_type (str): Model type - 'linear', 'ridge', 'lasso', 'elastic_net'
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        self.model = None
        self.hyperparameters = kwargs
        self.is_fitted = False
        
        # Create model instance
        self._create_model()
    
    def _create_model(self):
        """
        Create model instance based on model type
        
        Returns:
            None
        """
        model_configs = {
            'linear': {
                'class': LinearRegression,
                'default_params': {
                    'fit_intercept': True,
                    'n_jobs': None
                }
            },
            'ridge': {
                'class': Ridge,
                'default_params': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'random_state': 42
                }
            },
            'lasso': {
                'class': Lasso,
                'default_params': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'random_state': 42
                }
            },
            'elastic_net': {
                'class': ElasticNet,
                'default_params': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5,
                    'fit_intercept': True,
                    'random_state': 42
                }
            }
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        config = model_configs[self.model_type]
        
        # Merge default parameters and user parameters
        params = {**config['default_params'], **self.hyperparameters}
        
        # Create model instance
        self.model = config['class'](**params)
        
        print(f"Created {self.model_type} model with parameters: {params}")
    
    def fit(self, X, y):
        """
        Train the model
        
        Parameters:
            X: Training features
            y: Training target
            
        Returns:
            self: Returns self instance
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
            raise ValueError("Model not fitted. Please call fit method first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Parameters:
            X: Test features
            y: Test target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please call fit method first.")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform cross-validation
        
        Parameters:
            X: Training features
            y: Training target
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Cross-validation results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please call fit method first.")
        
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_folds': cv
            }
            
            print(f"Cross-validation completed. Mean score: {results['mean_cv_score']:.4f} (±{results['std_cv_score']:.4f})")
            return results
        except Exception as e:
            print(f"Cross-validation failed: {e}")
            raise
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
            X: Training features
            y: Training target
            param_grid (dict): Parameter grid for tuning
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Best parameters and score
        """
        if param_grid is None:
            # Default parameter grids for different model types
            if self.model_type == 'ridge':
                param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            elif self.model_type == 'lasso':
                param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            elif self.model_type == 'elastic_net':
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            else:
                param_grid = {}  # Linear regression has no hyperparameters to tune
        
        try:
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"Hyperparameter tuning completed. Best parameters: {results['best_params']}")
            return results
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            raise
    
    def get_feature_importance(self, X):
        """
        Get feature importance (coefficients for linear models)
        
        Parameters:
            X: Input features
            
        Returns:
            np.array: Feature importance (coefficients)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please call fit method first.")
        
        # For linear models, coefficients represent feature importance
        return self.model.coef_
    
    def create_models(self):
        """
        Create multiple linear regression models
        
        Returns:
            dict: Dictionary of model instances
        """
        models = {}
        
        model_types = ['linear', 'ridge', 'lasso', 'elastic_net']
        
        for model_type in model_types:
            models[model_type] = LinearRegressionModel(model_type)
        
        return models
    
    def get_hyperparameters(self, model_name=None, config=None):
        """
        Get hyperparameters for the specified model type
        
        Parameters:
            model_name (str): Model type name
            config (ConfigManager): Configuration manager
            
        Returns:
            dict: Hyperparameters dictionary
        """
        try:
            if config and hasattr(config, 'model_config') and model_name in config.model_config:
                return config.model_config[model_name]
        except:
            pass
        
        # Default parameters for different model types
        default_params = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elastic_net': {'alpha': 1.0, 'l1_ratio': 0.5}
        }
        
        return default_params.get(model_name, {})
    
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
            info['model_coefficients'] = self.model.coef_.tolist()
            info['model_intercept'] = self.model.intercept_
        
        return info


class LinearRegressionTrainer:
    """
    Linear Regression Model Trainer
    Handles complete training pipeline for linear regression models
    """
    
    def __init__(self, model_type='linear', hyperparameters=None):
        """
        Initialize trainer
        
        Parameters:
            model_type (str): Type of linear regression model
            hyperparameters (dict): Model hyperparameters
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.trained_model = None
        self.training_history = {}
    
    def create_model(self, hyperparameters=None):
        """
        Create a new model instance
        
        Parameters:
            hyperparameters (dict): Model hyperparameters
            
        Returns:
            LinearRegressionModel: Model instance
        """
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        self.model = LinearRegressionModel(self.model_type, **self.hyperparameters)
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   hyperparameter_tuning=True, param_grid=None, cv=5):
        """
        Train the model with optional hyperparameter tuning
        
        Parameters:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            param_grid (dict): Parameter grid for tuning
            cv (int): Cross-validation folds
            
        Returns:
            LinearRegressionModel: Trained model
        """
        if self.model is None:
            self.create_model()
        
        # Train the model
        self.trained_model = self.model.fit(X_train, y_train)
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning:
            tuning_results = self.model.hyperparameter_tuning(
                X_train, y_train, param_grid=param_grid, cv=cv
            )
            self.training_history['hyperparameter_tuning'] = tuning_results
        
        # Validation if validation set provided
        if X_val is not None and y_val is not None:
            val_metrics = self.model.evaluate(X_val, y_val)
            self.training_history['validation_metrics'] = val_metrics
        
        return self.trained_model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Parameters:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        metrics = self.trained_model.evaluate(X_test, y_test)
        
        # Cross-validation
        if hasattr(self.model, 'cross_validate'):
            cv_results = self.trained_model.cross_validate(X_test, y_test)
            metrics['cross_validation'] = cv_results
        
        return metrics
    
    def get_training_summary(self):
        """
        Get training summary
        
        Returns:
            dict: Training summary
        """
        summary = {
            'model_type': self.model_type,
            'trained': self.trained_model is not None,
            'training_history': self.training_history
        }
        
        if self.trained_model is not None:
            summary['model_info'] = self.trained_model.get_model_info()
        
        return summary