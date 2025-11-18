# -*- coding: utf-8 -*-
"""
Gradient Boosting Regression Model Module
Contains gradient boosting implementation for medical insurance cost prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingModel:
    """
    Gradient Boosting Regression Model Class
    Supports Gradient Boosting Regressor with extensive hyperparameter tuning
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model
        
        Parameters:
            **kwargs: Model hyperparameters
        """
        self.model = None
        self.hyperparameters = kwargs
        self.is_fitted = False
        
        # Create model instance
        self._create_model()
    
    def _create_model(self):
        """
        Create Gradient Boosting model instance
        
        Returns:
            None
        """
        # Default parameters for Gradient Boosting
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'max_features': None,
            'random_state': 42,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'tol': 1e-4
        }
        
        # Merge default parameters and user parameters
        params = {**default_params, **self.hyperparameters}
        
        # Create model instance
        self.model = GradientBoostingRegressor(**params)
        
        print(f"Created Gradient Boosting model with parameters: {params}")
    
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
            print("Gradient Boosting model training completed")
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
            # Default parameter grid for Gradient Boosting
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            }
        
        try:
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                verbose=1
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
        Get feature importance
        
        Parameters:
            X: Input features
            
        Returns:
            np.array: Feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please call fit method first.")
        
        # For gradient boosting, feature_importances_ represents feature importance
        return self.model.feature_importances_
    
    def create_models(self):
        """
        Create gradient boosting models
        
        Returns:
            dict: Dictionary containing gradient boosting model instances
        """
        models = {
            'gradient_boosting': GradientBoostingModel(**self.hyperparameters)
        }
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
            if config and hasattr(config, 'model_config') and 'gradient_boosting' in config.model_config:
                return config.model_config['gradient_boosting']
        except:
            pass
        
        # Default parameters for Gradient Boosting
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }

    def get_model_info(self):
        """
        Get model information
        
        Returns:
            dict: Model information
        """
        info = {
            'model_type': 'gradient_boosting',
            'is_fitted': self.is_fitted,
            'hyperparameters': self.hyperparameters
        }
        
        if self.is_fitted:
            info['feature_importance'] = self.model.feature_importances_.tolist()
            info['n_estimators'] = self.model.n_estimators
            
            # Add training score if available
            try:
                # We can't get training score directly, but we can store it during training
                if hasattr(self, '_training_score'):
                    info['training_score'] = self._training_score
                else:
                    info['training_score'] = None
            except:
                pass
        
        return info
    
    def plot_learning_curve(self, X_train, y_train, X_val=None, y_val=None):
        """
        Plot learning curve during training
        
        Parameters:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            None
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Please call fit method first.")
        
        # This method would typically be implemented with visualization
        # For now, we'll provide the structure
        train_scores = []
        val_scores = []
        
        # If validation set is provided
        if X_val is not None and y_val is not None:
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            
            train_scores = [np.sqrt(mean_squared_error(y_train, train_pred))]
            val_scores = [np.sqrt(mean_squared_error(y_val, val_pred))]
            
            print(f"Training RMSE: {train_scores[0]:.4f}")
            print(f"Validation RMSE: {val_scores[0]:.4f}")
        else:
            # Cross-validation approach
            cv_scores = self.cross_validate(X_train, y_train)
            print(f"Cross-validation RMSE: {np.sqrt(-cv_scores['mean_cv_score']):.4f}")
            print(f"Cross-validation RMSE std: {np.sqrt(cv_scores['std_cv_score']):.4f}")


class GradientBoostingTrainer:
    """
    Gradient Boosting Model Trainer
    Handles complete training pipeline for gradient boosting models
    """
    
    def __init__(self, hyperparameters=None):
        """
        Initialize trainer
        
        Parameters:
            hyperparameters (dict): Model hyperparameters
        """
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
            GradientBoostingModel: Model instance
        """
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        self.model = GradientBoostingModel(**self.hyperparameters)
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
            GradientBoostingModel: Trained model
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
        
        # Learning curve analysis
        if X_val is not None and y_val is not None:
            self.model.plot_learning_curve(X_train, y_train, X_val, y_val)
        else:
            self.model.plot_learning_curve(X_train, y_train)
        
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
            'model_type': 'gradient_boosting',
            'trained': self.trained_model is not None,
            'training_history': self.training_history
        }
        
        if self.trained_model is not None:
            summary['model_info'] = self.trained_model.get_model_info()
        
        return summary
    
    def get_feature_importance_analysis(self, feature_names):
        """
        Analyze feature importance with feature names
        
        Parameters:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance analysis
        """
        if self.trained_model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        importance = self.trained_model.get_feature_importance(None)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df