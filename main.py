# -*- coding: utf-8 -*-
"""
Medical Insurance Cost Prediction System
Main program integrating data loading, preprocessing, model training, evaluation and prediction
Supports linear regression and gradient boosting models with automated execution
"""

import pandas as pd
import numpy as np
import os
import logging
import pickle
import json
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from linear_model import LinearRegressionModel, LinearRegressionTrainer
from gradient_boosting_model import GradientBoostingModel, GradientBoostingTrainer
from config import ConfigManager

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_insurance_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MedicalInsuranceSystem:
    """
    Medical Insurance Cost Prediction System Main Class
    Integrates all module functions for complete prediction system
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the medical insurance prediction system
        
        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        # Initialize configuration manager
        self.config = config_manager or ConfigManager()
        
        # Initialize modules
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.linear_model = LinearRegressionModel()
        self.gradient_model = GradientBoostingModel()
        self.linear_trainer = LinearRegressionTrainer()
        self.gradient_trainer = GradientBoostingTrainer()
        
        # System state
        self.data = None
        self.processed_data = None
        self.trained_models = {}
        self.evaluation_results = {}
        self.feature_columns = None
        
        # Create output directories
        os.makedirs(self.config.output_config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_config.output_dir, "linear_results"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_config.output_dir, "gradient_results"), exist_ok=True)
        
        logger.info("Medical Insurance Prediction System initialized")
    
    def load_data(self):
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from: {self.config.data_config.data_path}")
            self.data = self.data_loader.load_from_csv(self.config.data_config.data_path)
            logger.info(f"Data loaded successfully, shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def preprocess_data(self):
        """
        Preprocess data including cleaning, encoding, and feature engineering
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_columns)
        """
        try:
            logger.info("Starting data preprocessing")
            
            # Basic data cleaning
            cleaned_data = self.data_preprocessor.basic_cleaning(self.data)
            logger.info(f"After basic cleaning, shape: {cleaned_data.shape}")
            
            # Handle outliers
            outlier_data = self.data_preprocessor.remove_outliers(cleaned_data)
            logger.info(f"After outlier handling, shape: {outlier_data.shape}")
            
            # Encode categorical features
            encoded_data = self.data_preprocessor.encode_categorical_features(outlier_data)
            logger.info("Categorical feature encoding completed")
            
            # Feature engineering
            engineered_data = self.data_preprocessor.feature_engineering(encoded_data)
            logger.info("Feature engineering completed")
            
            # Get feature columns
            feature_columns = self.config.feature_config.categorical_features + \
                             self.config.feature_config.numerical_features
            
            # Feature scaling
            if self.config.data_config.standardize_features:
                feature_data = engineered_data[feature_columns]
                scaled_data = self.data_preprocessor.scale_features(feature_data)
                self.processed_data = pd.concat([
                    scaled_data,
                    engineered_data['annual_premium']
                ], axis=1)
                logger.info("Feature scaling completed")
            else:
                self.processed_data = engineered_data[feature_columns + ['annual_premium']]
            
            # Split train and test sets
            X_train, X_test, y_train, y_test = self.data_preprocessor.split_data(
                self.processed_data,
                test_size=self.config.data_config.test_size,
                random_state=self.config.data_config.random_state
            )
            
            self.feature_columns = feature_columns
            
            logger.info(f"Data splitting completed - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, feature_columns
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train both linear regression and gradient boosting models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        """
        try:
            logger.info("Starting model training")
            
            # Train linear regression models
            logger.info("Training linear regression models")
            linear_models = self.linear_model.create_models()
            
            for model_name, model in linear_models.items():
                try:
                    logger.info(f"Training {model_name} regression model")
                    
                    # Get hyperparameters for this model
                    params = self.linear_model.get_hyperparameters(model_name, self.config)
                    
                    # Update model with hyperparameters
                    if params:
                        model.hyperparameters.update(params)
                    
                    # Create new trainer for this specific model type
                    specific_trainer = LinearRegressionTrainer(model_type=model_name, hyperparameters=params)
                    
                    # Train model
                    trained_model = specific_trainer.train_model(X_train, y_train)
                    
                    # Make predictions
                    y_pred = trained_model.predict(X_test)
                    
                    # Store results
                    self.trained_models[f"linear_{model_name}"] = {
                        'model': trained_model,
                        'predictions': y_pred,
                        'true_values': y_test,
                        'model_type': 'linear'
                    }
                    
                    logger.info(f"Linear {model_name} model training completed")
                    
                except Exception as e:
                    logger.error(f"Linear {model_name} model training failed: {e}")
                    continue
            
            # Train gradient boosting model
            logger.info("Training gradient boosting model")
            try:
                # Create gradient boosting model
                gb_models = self.gradient_model.create_models()
                gb_model = gb_models['gradient_boosting']
                
                # Get hyperparameters
                params = self.gradient_model.get_hyperparameters(config=self.config)
                
                # Update model hyperparameters
                if params:
                    gb_model.hyperparameters.update(params)
                
                # Create specific trainer for gradient boosting
                specific_trainer = GradientBoostingTrainer(hyperparameters=params)
                
                # Train model
                trained_gb_model = specific_trainer.train_model(X_train, y_train)
                
                y_pred_gb = trained_gb_model.predict(X_test)
                
                self.trained_models["gradient_boosting"] = {
                    'model': trained_gb_model,
                    'predictions': y_pred_gb,
                    'true_values': y_test,
                    'model_type': 'gradient'
                }
                
                logger.info("Gradient boosting model training completed")
                
            except Exception as e:
                logger.error(f"Gradient boosting model training failed: {e}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def evaluate_models(self):
        """
        Evaluate all trained models and generate visualizations
        
        Returns:
            dict: Evaluation results for all models
        """
        try:
            logger.info("Starting model evaluation")
            
            from evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            
            evaluation_results = {}
            
            for model_name, model_info in self.trained_models.items():
                logger.info(f"Evaluating {model_name} model")
                
                # Calculate metrics
                y_true = model_info['true_values']
                y_pred = model_info['predictions']
                
                metrics = evaluator.calculate_metrics(y_true, y_pred)
                
                # Store results
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'true_values': y_true,
                    'residuals': y_true - y_pred,
                    'model_type': model_info['model_type']
                }
                
                # Generate visualizations
                self._create_model_visualizations(model_name, y_true, y_pred, metrics, model_info['model_type'])
                
                # Save numerical results
                self._save_numerical_results(model_name, metrics, model_info['model_type'])
            
            self.evaluation_results = evaluation_results
            logger.info("Model evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _create_model_visualizations(self, model_name, y_true, y_pred, metrics, model_type):
        """
        Create comprehensive visualizations for a model
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
            model_type: Type of model ('linear' or 'gradient')
        """
        try:
            # Determine output directory
            if model_type == 'linear':
                output_dir = self.config.output_config.linear_result_dir
            else:
                output_dir = self.config.output_config.gradient_result_dir
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{model_name.replace("_", " ").title()} Model Evaluation', fontsize=16, fontweight='bold')
            
            # 1. Actual vs Predicted scatter plot
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=30)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values', fontsize=12)
            axes[0, 0].set_ylabel('Predicted Values', fontsize=12)
            axes[0, 0].set_title('Actual vs Predicted Values', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add R² annotation
            axes[0, 0].text(0.05, 0.95, f'R² = {metrics["R2"]:.4f}', 
                           transform=axes[0, 0].transAxes, fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 2. Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
            axes[0, 1].set_ylabel('Residuals', fontsize=12)
            axes[0, 1].set_title('Residuals Plot', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Distribution of residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Distribution of Residuals', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[1, 0].text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                           transform=axes[1, 0].transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 4. Metrics summary
            axes[1, 1].axis('off')
            metrics_text = f"""
            Model Performance Metrics:
            
            Mean Squared Error (MSE): {metrics['MSE']:.4f}
            Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}
            Mean Absolute Error (MAE): {metrics['MAE']:.4f}
            R² Score: {metrics['R2']:.4f}
            
            Model Type: {model_type.replace('_', ' ').title()}
            """
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                           verticalalignment='center')
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{model_name}_evaluation.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations for {model_name}: {e}")
    
    def _save_numerical_results(self, model_name, metrics, model_type):
        """
        Save numerical evaluation results to files
        
        Args:
            model_name: Name of the model
            metrics: Evaluation metrics dictionary
            model_type: Type of model
        """
        try:
            # Determine output directory
            if model_type == 'linear':
                output_dir = self.config.output_config.linear_result_dir
            else:
                output_dir = self.config.output_config.gradient_result_dir
            
            # Save metrics as JSON
            json_filename = f"{model_name}_metrics.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Save metrics as CSV
            csv_filename = f"{model_name}_metrics.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            pd.DataFrame([metrics]).to_csv(csv_path, index=False)
            
            logger.info(f"Numerical results saved for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save numerical results for {model_name}: {e}")
    
    def create_model_comparison(self):
        """
        Create comprehensive model comparison visualization
        
        Returns:
            str: Path to comparison plot
        """
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results available for comparison")
                return None
            
            # Prepare data for comparison
            model_names = []
            metrics_data = {metric: [] for metric in ['MSE', 'RMSE', 'MAE', 'R2']}
            
            for model_name, results in self.evaluation_results.items():
                model_names.append(model_name.replace('_', ' ').title())
                for metric in metrics_data:
                    metrics_data[metric].append(results['metrics'][metric])
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # Colors for different model types
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum']
            
            # MSE Comparison
            bars1 = axes[0, 0].bar(model_names, metrics_data['MSE'], color=colors[:len(model_names)])
            axes[0, 0].set_title('Mean Squared Error (MSE)', fontsize=14)
            axes[0, 0].set_ylabel('MSE', fontsize=12)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, metrics_data['MSE']):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['MSE'])*0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            # RMSE Comparison
            bars2 = axes[0, 1].bar(model_names, metrics_data['RMSE'], color=colors[:len(model_names)])
            axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=14)
            axes[0, 1].set_ylabel('RMSE', fontsize=12)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, metrics_data['RMSE']):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['RMSE'])*0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            # MAE Comparison
            bars3 = axes[1, 0].bar(model_names, metrics_data['MAE'], color=colors[:len(model_names)])
            axes[1, 0].set_title('Mean Absolute Error (MAE)', fontsize=14)
            axes[1, 0].set_ylabel('MAE', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, metrics_data['MAE']):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['MAE'])*0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            # R² Comparison
            bars4 = axes[1, 1].bar(model_names, metrics_data['R2'], color=colors[:len(model_names)])
            axes[1, 1].set_title('R² Score', fontsize=14)
            axes[1, 1].set_ylabel('R² Score', fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, metrics_data['R2']):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_data['R2'])*0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            # Save comparison plot
            comparison_path = os.path.join(self.config.output_config.output_dir, 'model_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved: {comparison_path}")
            
            # Save comparison data
            comparison_data = {
                'model_names': model_names,
                'metrics': metrics_data
            }
            
            comparison_json_path = os.path.join(self.config.output_config.output_dir, 'model_comparison.json')
            with open(comparison_json_path, 'w') as f:
                json.dump(comparison_data, f, indent=4)
            
            return comparison_path
            
        except Exception as e:
            logger.error(f"Failed to create model comparison: {e}")
            return None
    
    def print_summary_report(self):
        """
        Print comprehensive summary report of the analysis
        """
        print("\n" + "=" * 80)
        print("MEDICAL INSURANCE COST PREDICTION - ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Data summary
        print(f"\nDATA SUMMARY:")
        print(f"- Dataset shape: {self.data.shape}")
        print(f"- Features used: {len(self.feature_columns)}")
        print(f"- Feature columns: {', '.join(self.feature_columns)}")
        
        # Models summary
        print(f"\nMODELS TRAINED:")
        for model_name in self.trained_models.keys():
            print(f"- {model_name.replace('_', ' ').title()}")
        
        # Performance summary
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  MSE (Mean Squared Error): {metrics['MSE']:.4f}")
            print(f"  RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
            print(f"  MAE (Mean Absolute Error): {metrics['MAE']:.4f}")
            print(f"  R² (Coefficient of Determination): {metrics['R2']:.4f}")
            
            if metrics['R2'] > best_r2:
                best_r2 = metrics['R2']
                best_model = model_name
        
        # Best model
        if best_model:
            print(f"\nBEST PERFORMING MODEL:")
            print(f"- {best_model.replace('_', ' ').title()} (R² = {best_r2:.4f})")
        
        # Output locations
        print(f"\nOUTPUT LOCATIONS:")
        print(f"- Linear regression results: {os.path.join(self.config.output_config.output_dir, 'linear_results')}")
        print(f"- Gradient boosting results: {os.path.join(self.config.output_config.output_dir, 'gradient_results')}")
        print(f"- Model comparison: {os.path.join(self.config.output_config.output_dir, 'model_comparison.png')}")
        
        print("\n" + "=" * 80)
    
    def run_automated_pipeline(self, **model_params):
        """
        Run the complete automated prediction pipeline
        
        Args:
            **model_params: Additional model hyperparameters to override defaults
            
        Returns:
            dict: Complete analysis results
        """
        try:
            logger.info("Starting automated medical insurance cost prediction pipeline")
            
            start_time = datetime.now()
            
            # Update model parameters if provided
            if model_params:
                for param_name, param_value in model_params.items():
                    if param_name in ['learning_rate', 'n_estimators', 'max_depth', 'alpha', 'ridge_alpha', 'lasso_alpha']:
                        if param_name == 'learning_rate':
                            self.config.update_config('model', gb_learning_rate=param_value)
                        elif param_name == 'n_estimators':
                            self.config.update_config('model', gb_n_estimators=param_value)
                        elif param_name == 'max_depth':
                            self.config.update_config('model', gb_max_depth=param_value)
                        elif param_name == 'alpha':
                            self.config.update_config('model', regularization_alpha=param_value)
                        elif param_name == 'ridge_alpha':
                            self.config.update_config('model', ridge_alpha=param_value)
                        elif param_name == 'lasso_alpha':
                            self.config.update_config('model', lasso_alpha=param_value)
            
            # Execute pipeline steps
            self.load_data()
            X_train, X_test, y_train, y_test, feature_columns = self.preprocess_data()
            self.train_models(X_train, y_train, X_test, y_test)
            evaluation_results = self.evaluate_models()
            comparison_plot_path = self.create_model_comparison()
            
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            # Generate comprehensive results
            results = {
                'execution_time': str(execution_time),
                'data_shape': self.data.shape,
                'feature_count': len(feature_columns),
                'feature_columns': feature_columns,
                'trained_models': list(self.trained_models.keys()),
                'evaluation_results': evaluation_results,
                'best_model': max(evaluation_results.items(), key=lambda x: x[1]['metrics']['R2'])[0],
                'comparison_plot_path': comparison_plot_path
            }
            
            # Print summary report
            self.print_summary_report()
            
            logger.info(f"Automated pipeline completed in {execution_time}")
            
            return results
            
        except Exception as e:
            logger.error(f"Automated pipeline failed: {e}")
            raise


def main():
    """
    Main function - Automated execution
    Demonstrates how to use the medical insurance prediction system
    """
    print("=" * 80)
    print("MEDICAL INSURANCE COST PREDICTION SYSTEM")
    print("Automated Analysis with Linear Regression and Gradient Boosting")
    print("=" * 80)
    
    try:
        # Create system instance
        system = MedicalInsuranceSystem()
        
        # Define custom model parameters (optional)
        model_parameters = {
            'learning_rate': 0.1,        # Gradient boosting learning rate
            'n_estimators': 100,         # Number of estimators
            'max_depth': 3,              # Maximum depth
            'ridge_alpha': 1.0,          # Ridge regression regularization
            'lasso_alpha': 0.1           # Lasso regression regularization
        }
        
        print("\nStarting automated analysis...")
        print(f"Model parameters: {model_parameters}")
        
        # Run the complete automated pipeline
        results = system.run_automated_pipeline(**model_parameters)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Execution time: {results['execution_time']}")
        print(f"Best model: {results['best_model'].replace('_', ' ').title()}")
        
        # Display final summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE - CHECK OUTPUT DIRECTORIES")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        print(f"Program execution failed: {e}")
        raise


if __name__ == "__main__":
    main()