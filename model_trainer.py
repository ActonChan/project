# -*- coding: utf-8 -*-
"""
Model Training Module
Handle training process and parameter optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model_core import ModelCore
import time
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model Trainer Class
    Manage model training, validation, and hyperparameter optimization processes
    """
    
    def __init__(self, model_type='gradient_boosting'):
        """
        Initialize model trainer
        
        Parameters:
            model_type (str): Model type
        """
        self.model_type = model_type
        self.model = None
        self.training_history = []
        self.best_params = None
        self.best_score = None
        self.is_trained = False
        
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   hyperparameters=None, early_stopping=True, verbose=True):
        """
        Train model
        
        Parameters:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            hyperparameters (dict): Hyperparameter configuration
            early_stopping (bool): Whether to use early stopping mechanism
            verbose (bool): Whether to display training progress
            
        Returns:
            ModelCore: Trained model
        """
        # Create model instance
        self.model = ModelCore(model_type=self.model_type, **(hyperparameters or {}))
        
        start_time = time.time()
        
        if verbose:
            print(f"Starting training {self.model_type} model...")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Validation set evaluation (if provided)
        if X_val is not None and y_val is not None:
            train_score = self.model.evaluate(X_train, y_train, 'rmse')
            val_score = self.model.evaluate(X_val, y_val, 'rmse')
            
            training_record = {
                'timestamp': time.time(),
                'duration': time.time() - start_time,
                'train_rmse': train_score,
                'val_rmse': val_score,
                'hyperparameters': hyperparameters or {}
            }
            
            if verbose:
                print(f"Training completed, duration: {training_record['duration']:.2f}s")
                print(f"Training RMSE: {train_score:.4f}")
                print(f"Validation RMSE: {val_score:.4f}")
        else:
            # Use cross-validation evaluation
            cv_results = self.model.cross_validate(X_train, y_train, cv=5)
            
            training_record = {
                'timestamp': time.time(),
                'duration': time.time() - start_time,
                'cv_mean_score': cv_results['mean_score'],
                'cv_std_score': cv_results['std_score'],
                'hyperparameters': hyperparameters or {}
            }
            
            if verbose:
                print(f"Training completed, duration: {training_record['duration']:.2f}s")
                print(f"Cross-validation RMSE: {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f})")
        
        self.training_history.append(training_record)
        self.is_trained = True
        
        return self.model
    
    def hyperparameter_optimization(self, X_train, y_train, param_grid=None, 
                                  method='grid', cv=5, scoring='neg_mean_squared_error', 
                                  n_iter=50, verbose=1):
        """
        Hyperparameter optimization
        
        Parameters:
            X_train: Training features
            y_train: Training targets
            param_grid (dict): Parameter grid
            method (str): Optimization method ('grid' or 'random')
            cv (int): Cross-validation folds
            scoring (str): Evaluation metric
            n_iter (int): Random search iterations
            verbose (int): Output verbosity level
            
        Returns:
            tuple: (best parameters, best model)
        """
        if param_grid is None:
            if self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'linear_regression':
                param_grid = {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                }
            else:
                param_grid = {}
        
        print(f"Starting {method} search for hyperparameters...")
        
        # Create base model
        base_model = ModelCore(model_type=self.model_type)
        
        if method == 'grid':
            search = GridSearchCV(
                base_model.sklearn_model, param_grid, cv=cv, scoring=scoring, 
                verbose=verbose, n_jobs=-1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                base_model.sklearn_model, param_grid, n_iter=n_iter, cv=cv, 
                scoring=scoring, verbose=verbose, n_jobs=-1, random_state=42
            )
        else:
            raise ValueError(f"Unsupported search method: {method}")
        
        start_time = time.time()
        
        # Fit search
        search.fit(X_train, y_train)
        
        duration = time.time() - start_time
        
        print(f"Hyperparameter optimization completed, duration: {duration:.2f}s")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {-search.best_score_:.4f}")
        
        # Create final model with best parameters
        self.model = ModelCore(model_type=self.model_type, **search.best_params_)
        self.model.fit(X_train, y_train)
        
        optimization_record = {
            'timestamp': time.time(),
            'duration': duration,
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'method': method,
            'cv': cv
        }
        
        self.training_history.append(optimization_record)
        self.is_trained = True
        
        return search.best_params_, self.model
    
    def ensemble_training(self, X_train, y_train, models_config=None, voting='average'):
        """
        Ensemble learning training
        
        Parameters:
            X_train: Training features
            y_train: Training targets
            models_config (list): Model configuration list
            voting (str): Voting strategy ('average', 'weighted')
            
        Returns:
            dict: Ensemble model training results
        """
        if models_config is None:
            models_config = [
                'gradient_boosting',
                'random_forest',
                'ridge',
                'lasso'
            ]
        
        print(f"Starting ensemble learning training with {len(models_config)} models...")
        print(f"Model list: {models_config}")
        
        ensemble_models = []
        ensemble_scores = []
        
        for i, model_type in enumerate(models_config):
            print(f"Training model {i+1}: {model_type}")
            
            try:
                # Create and train model
                model = ModelCore(model_type=model_type)
                model.fit(X_train, y_train)
                
                # Evaluate model
                score = model.evaluate(X_train, y_train, 'rmse')
                
                ensemble_models.append((model_type, model))
                ensemble_scores.append(score)
                
                print(f"  {model_type} training completed, RMSE: {score:.4f}")
                
            except Exception as e:
                print(f"  {model_type} training failed: {str(e)}")
                continue
        
        if not ensemble_models:
            raise ValueError("All models failed to train")
        
        # Simple ensemble strategy: average prediction
        self.ensemble_models = ensemble_models
        self.ensemble_scores = ensemble_scores
        
        print(f"Ensemble training completed, {len(ensemble_models)} models trained successfully")
        
        ensemble_result = {
            'models': ensemble_models,
            'scores': ensemble_scores,
            'voting': voting,
            'best_model': min(ensemble_models, key=lambda x: x[1]),
            'average_score': np.mean(ensemble_scores)
        }
        
        return ensemble_result
    
    def evaluate_multiple_metrics(self, X, y):
        """
        多种指标评估
        
        参数:
            X: 特征数据
            y: 目标数据
            
        返回:
            dict: 多种评估指标结果
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
            'MAPE': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        print("模型评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def get_training_summary(self):
        """
        获取训练摘要
        
        返回:
            pd.DataFrame: 训练历史摘要
        """
        if not self.training_history:
            return pd.DataFrame()
        
        # 转换为DataFrame
        summary_df = pd.DataFrame(self.training_history)
        
        # 添加模型信息
        summary_df['model_type'] = self.model_type
        summary_df['is_trained'] = self.is_trained
        
        return summary_df
    
    def evaluate_models(self, models, X_test, y_test, metrics=['rmse', 'r2', 'mae']):
        """
        Evaluate performance of multiple models
        
        Parameters:
            models (list): Model list
            X_test: Test features
            y_test: Test targets
            metrics (list): Evaluation metrics list
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        for model_name, model in models:
            try:
                predictions = model.predict(X_test)
                
                model_results = {}
                
                if 'rmse' in metrics:
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    model_results['rmse'] = rmse
                
                if 'r2' in metrics:
                    r2 = r2_score(y_test, predictions)
                    model_results['r2'] = r2
                
                if 'mae' in metrics:
                    mae = mean_absolute_error(y_test, predictions)
                    model_results['mae'] = mae
                
                results[model_name] = model_results
                
                print(f"{model_name} evaluation results:")
                for metric, value in model_results.items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"{model_name} evaluation failed: {str(e)}")
                results[model_name] = None
        
        return results

    def plot_training_curves(self, save_path=None):
        """
        Plot training curves
        
        Parameters:
            save_path (str): Save path (optional)
        """
        import matplotlib.pyplot as plt
        
        if not self.training_history:
            print("No training history available for plotting")
            return
        
        history_df = pd.DataFrame(self.training_history)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training time
        axes[0].plot(range(len(history_df)), history_df['duration'])
        axes[0].set_title('Training Time Changes')
        axes[0].set_xlabel('Training Epoch')
        axes[0].set_ylabel('Training Time (seconds)')
        axes[0].grid(True)
        
        # Validation scores (if available)
        val_columns = [col for col in history_df.columns if 'val_' in col or 'cv_' in col]
        for col in val_columns:
            axes[1].plot(range(len(history_df)), history_df[col], label=col)
        
        axes[1].set_title('Model Performance Changes')
        axes[1].set_xlabel('Training Epoch')
        axes[1].set_ylabel('Performance Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()


class AdvancedModelTrainer(ModelTrainer):
    """
    Advanced Model Trainer
    Extended base trainer functionality with more advanced features
    """
    
    def __init__(self, model_type='gradient_boosting'):
        super().__init__(model_type)
        self.cv_scores_history = []
        self.feature_importance_history = []
    
    def progressive_training(self, X_train, y_train, n_splits=5, feature_increment=5):
        """
        Progressive training
        Gradually increase the number of features to train model and observe performance changes
        
        Parameters:
            X_train: Feature data
            y_train: Target data
            n_splits (int): Number of cross-validation folds
            feature_increment (int): Feature increment
            
        Returns:
            dict: Progressive training results
        """
        n_features = X_train.shape[1]
        feature_counts = range(feature_increment, n_features + 1, feature_increment)
        
        progressive_results = []
        
        print(f"Starting progressive training, feature increment: {feature_increment}")
        
        for n_feat in feature_counts:
            # Select first n features
            X_subset = X_train.iloc[:, :n_feat]
            
            # Train model
            model = ModelCore(model_type=self.model_type)
            cv_scores = model.cross_validate(X_subset, y_train, cv=n_splits)
            
            progressive_results.append({
                'n_features': n_feat,
                'cv_mean_score': cv_scores['mean_score'],
                'cv_std_score': cv_scores['std_score']
            })
            
            print(f"Number of features: {n_feat}, cross-validation RMSE: {cv_scores['mean_score']:.4f}")
        
        # Find best feature count
        best_result = min(progressive_results, key=lambda x: x['cv_mean_score'])
        best_n_features = best_result['n_features']
        
        print(f"Best number of features: {best_n_features}")
        
        return {
            'progressive_results': progressive_results,
            'best_n_features': best_n_features,
            'best_score': best_result['cv_mean_score']
        }
    
    def auto_ml_training(self, X_train, y_train, time_limit=3600, metric='rmse'):
        """
        Automated machine learning training
        Automatically search for best model and hyperparameters
        
        Parameters:
            X_train: Feature data
            y_train: Target data
            time_limit (int): Time limit (seconds)
            metric (str): Optimization metric
            
        Returns:
            dict: AutoML results
        """
        from datetime import datetime, timedelta
        
        print(f"Starting automated machine learning training, time limit: {time_limit}s")
        
        # Define models to try
        models_to_try = [
            'gradient_boosting', 'random_forest', 'xgboost', 
            'ridge', 'lasso', 'elastic_net'
        ]
        
        auto_ml_results = {}
        start_time = datetime.now()
        
        for model_type in models_to_try:
            # Check remaining time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > time_limit:
                print(f"Time limit reached, stopping training of remaining models")
                break
            
            print(f"Training model: {model_type}")
            
            try:
                # Create trainer
                trainer = ModelTrainer(model_type)
                
                # Simplified hyperparameter search
                param_grids = {
                    'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]},
                    'random_forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
                    'xgboost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]},
                    'ridge': {'alpha': [1.0, 10.0]},
                    'lasso': {'alpha': [1.0, 10.0]},
                    'elastic_net': {'alpha': [1.0, 10.0], 'l1_ratio': [0.5, 0.7]}
                }
                
                # Hyperparameter optimization
                if model_type in param_grids:
                    optimization_result = trainer.hyperparameter_optimization(
                        X_train, y_train, method='grid', 
                        param_grid=param_grids[model_type],
                        cv=3  # Reduce folds to save time
                    )
                    
                    # Retrain with best parameters
                    best_model = trainer.train_model(
                        X_train, y_train, 
                        hyperparameters=optimization_result[0],
                        verbose=False
                    )
                    
                    auto_ml_results[model_type] = {
                        'model': best_model,
                        'best_params': optimization_result[0],
                        'best_score': optimization_result[1],
                        'training_time': trainer.training_history[0]['duration']
                    }
                    
                else:
                    # Direct model training
                    model = trainer.train_model(X_train, y_train, verbose=False)
                    score = model.evaluate(X_train, y_train, metric)
                    
                    auto_ml_results[model_type] = {
                        'model': model,
                        'score': score,
                        'training_time': trainer.training_history[0]['duration']
                    }
                
            except Exception as e:
                print(f"Model {model_type} training failed: {e}")
                continue
        
        # Select best model
        if auto_ml_results:
            best_model_type = min(auto_ml_results.keys(), 
                                key=lambda x: auto_ml_results[x].get('best_score', 
                                                                   auto_ml_results[x].get('score', float('inf'))))
            
            print(f"Best model: {best_model_type}")
            
            return {
                'results': auto_ml_results,
                'best_model_type': best_model_type,
                'best_model': auto_ml_results[best_model_type]['model'],
                'best_score': auto_ml_results[best_model_type].get('best_score', 
                                                                 auto_ml_results[best_model_type].get('score'))
            }
        
        return {'results': auto_ml_results}