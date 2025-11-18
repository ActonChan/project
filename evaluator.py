# -*- coding: utf-8 -*-
"""
Evaluation and Prediction Module
Implement model evaluation metrics and prediction functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           mean_absolute_percentage_error, explained_variance_score)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set font configuration for plots
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """
    Model Evaluator Class
    Responsible for model performance evaluation, prediction, and result analysis
    """
    
    def __init__(self, feature_columns=None, categorical_features=None):
        """
        Initialize evaluator
        
        Parameters:
            feature_columns (list): Feature column names list
            categorical_features (list): Categorical features list
        """
        self.feature_columns = feature_columns or []
        self.categorical_features = categorical_features or []
        self.prediction_history = []
        self.evaluation_history = []
        
    def calculate_metrics(self, y_true, y_pred, sample_weight=None):
        """
        Calculate multiple evaluation metrics
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights
            
        Returns:
            dict: Evaluation metrics dictionary
        """
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
            'MAE': mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'Explained_Variance': explained_variance_score(y_true, y_pred)
        }
        
        return metrics
    
    def detailed_evaluation(self, model, X_test, y_test, feature_names=None, save_plots=True):
        """
        Detailed model evaluation
        
        Parameters:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            feature_names (list): Feature names list
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Detailed evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Save evaluation history
        evaluation_record = {
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'n_samples': len(y_test),
            'feature_names': feature_names or self.feature_columns
        }
        self.evaluation_history.append(evaluation_record)
        
        # Create visualizations
        if save_plots:
            self._create_evaluation_plots(y_test, y_pred, metrics, save_path='.')
        
        # Create evaluation report
        report = {
            'metrics': metrics,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'residuals': y_test - y_pred
            },
            'sample_statistics': {
                'mean_true': np.mean(y_test),
                'std_true': np.std(y_test),
                'mean_pred': np.mean(y_pred),
                'std_pred': np.std(y_pred),
                'correlation': np.corrcoef(y_test, y_pred)[0, 1]
            }
        }
        
        return report
    
    def _create_evaluation_plots(self, y_true, y_pred, metrics, save_path='.'):
        """
        Create evaluation plots
        
        Parameters:
            y_true: True values
            y_pred: Predicted values
            metrics: Evaluation metrics
            save_path: Save path
        """
        # Set font configuration for plots
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        # 1. True vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('True vs Predicted Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual distribution histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics display
        axes[1, 1].axis('off')
        metrics_text = []
        for metric, value in metrics.items():
            if isinstance(value, float):
                metrics_text.append(f'{metric}: {value:.4f}')
            else:
                metrics_text.append(f'{metric}: {value}')
        
        axes[1, 1].text(0.1, 0.9, 'Evaluation Metrics', fontsize=14, fontweight='bold', 
                       transform=axes[1, 1].transAxes)
        for i, text in enumerate(metrics_text):
            axes[1, 1].text(0.1, 0.7 - i*0.1, text, fontsize=12, 
                           transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, 'model_evaluation_dashboard.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to: {plot_path}")
        
        return plot_path
    
    def _plot_feature_importance(self, model, save_path='.'):
        """
        Plot feature importance
        
        Parameters:
            model: Trained model
            save_path: Save path
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance")
            return
        
        feature_names = self.feature_columns if self.feature_columns else [f'Feature_{i}' for i in range(len(model.feature_importances_))]
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save plot
        importance_path = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {importance_path}")
        return feature_importance_df
    
    def compare_models(self, models_dict, X_test, y_test, model_names=None):
        """
        Compare performance of multiple models
        
        Parameters:
            models_dict: Model dictionary {'Model Name': model object}
            X_test: Test features
            y_test: Test targets
            model_names: Custom model names list
            
        Returns:
            dict: Comparison results dictionary
        """
        if not models_dict:
            raise ValueError("Model dictionary cannot be empty")
        
        comparison_results = {}
        model_names = list(models_dict.keys()) if not model_names else model_names
        
        print("Starting model comparison...")
        
        for i, (model_key, model) in enumerate(models_dict.items()):
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                
                # Store results
                model_name = model_names[i] if i < len(model_names) else model_key
                comparison_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                print(f"{model_name} evaluation completed")
                
            except Exception as e:
                print(f"Error evaluating model {model_key}: {str(e)}")
                continue
        
        # Create comparison plots
        if len(comparison_results) > 1:
            self._create_model_comparison_plots(comparison_results)
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(comparison_results)
        
        print("Model comparison completed")
        
        return {
            'results': comparison_results,
            'report': comparison_report
        }
    
    def _create_model_comparison_plots(self, comparison_results):
        """
        Create model comparison plots
        
        Parameters:
            comparison_results: Comparison results dictionary
        """
        # Extract metrics for all models
        model_names = list(comparison_results.keys())
        metrics_data = {metric: [] for metric in ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']}
        
        for model_name in model_names:
            metrics = comparison_results[model_name]['metrics']
            for metric in metrics_data:
                metrics_data[metric].append(metrics[metric])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics_data.items()):
            if i < 6:  # Only plot first 6 metrics
                row = i // 3
                col = i % 3
                axes[row, col].bar(model_names, values, alpha=0.7)
                axes[row, col].set_title(f'{metric} Comparison')
                axes[row, col].set_ylabel(metric)
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplots
        if len(metrics_data) < 6:
            for i in range(len(metrics_data), 6):
                row = i // 3
                col = i % 3
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save plot
        comparison_plot_path = 'model_comparison.png'
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model comparison plot saved to: {comparison_plot_path}")
        
        return comparison_results
    
    def predict_batch(self, model, X_batch, batch_size=32):
        """
        Batch prediction
        
        Parameters:
            model: Trained model
            X_batch: Data to predict
            batch_size: Batch size
            
        Returns:
            np.array: Prediction results
        """
        n_samples = len(X_batch)
        predictions = np.zeros(n_samples)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_data = X_batch.iloc[i:end_idx]
            batch_pred = model.predict(batch_data)
            predictions[i:end_idx] = batch_pred
        
        return predictions
    
    def predict_single(self, model, user_input_dict, preprocessor=None):
        """
        Single sample prediction
        
        Parameters:
            model: Trained model
            user_input_dict (dict): User input data
            preprocessor: Data preprocessor
            
        Returns:
            float: Prediction result
        """
        # Convert to DataFrame
        input_df = pd.DataFrame([user_input_dict])
        
        # Preprocess if preprocessor is provided
        if preprocessor is not None:
            # Encode categorical features
            for col in preprocessor.categorical_features:
                if col in input_df.columns and col in preprocessor.label_encoders:
                    le = preprocessor.label_encoders[col]
                    if input_df[col].iloc[0] in le.classes_:
                        input_df[col] = le.transform(input_df[col])
                    else:
                        # Handle unseen labels
                        input_df[col] = le.classes_[0]
            
            # Feature engineering
            input_df = preprocessor.feature_engineering(input_df, is_training=False)
            
            # Ensure feature order
            if preprocessor.feature_columns:
                for col in preprocessor.feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[preprocessor.feature_columns]
            
            # Scale features
            if preprocessor.scaler is not None:
                input_scaled = preprocessor.scaler.transform(input_df)
                input_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Save prediction history
        prediction_record = {
            'timestamp': pd.Timestamp.now(),
            'input': user_input_dict,
            'prediction': prediction
        }
        self.prediction_history.append(prediction_record)
        
        return prediction
    
    def generate_report(self, evaluation_data, feature_importance=None, save_path='evaluation_report.txt'):
        """
        Generate evaluation report
        
        Parameters:
            evaluation_data (dict): Evaluation data
            feature_importance (pd.DataFrame): Feature importance
            save_path (str): Report save path
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Medical Insurance Cost Prediction Model Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {pd.Timestamp.now()}")
        report_lines.append("")
        
        # Model performance metrics
        report_lines.append("1. Model Performance Metrics")
        report_lines.append("-" * 30)
        metrics = evaluation_data['metrics']
        for metric, value in metrics.items():
            report_lines.append(f"{metric:15}: {value:.4f}")
        report_lines.append("")
        
        # Sample statistics
        report_lines.append("2. Sample Statistics")
        report_lines.append("-" * 30)
        stats = evaluation_data['sample_statistics']
        for stat, value in stats.items():
            report_lines.append(f"{stat:15}: {value:.4f}")
        report_lines.append("")
        
        # Feature importance
        if feature_importance is not None:
            report_lines.append("3. Feature Importance (Top 10)")
            report_lines.append("-" * 30)
            for idx, row in feature_importance.head(10).iterrows():
                report_lines.append(f"{row['Feature']:20}: {row['Importance']:.4f}")
            report_lines.append("")
        
        # Model recommendations
        report_lines.append("4. Model Evaluation Recommendations")
        report_lines.append("-" * 30)
        
        r2_score = metrics['R2']
        if r2_score > 0.8:
            report_lines.append("• Model fit is very good with high R² score")
        elif r2_score > 0.6:
            report_lines.append("• Model fit is good, can be further optimized")
        else:
            report_lines.append("• Model fit is average, suggest adjusting features or algorithms")
        
        rmse = metrics['RMSE']
        mae = metrics['MAE']
        if mae < rmse * 0.5:
            report_lines.append("• MAE is relatively small compared to RMSE, indicating good robustness to outliers")
        
        report_lines.append("")
        report_lines.append("5. Improvement Suggestions")
        report_lines.append("-" * 30)
        report_lines.append("• Consider adding more relevant features")
        report_lines.append("• Try different machine learning algorithms")
        report_lines.append("• Perform more detailed feature engineering")
        report_lines.append("• Adjust model hyperparameters")
        
        # Save report
        report_content = "\n".join(report_lines)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Evaluation report saved to: {save_path}")
        return report_content
    
    def get_prediction_history(self):
        """
        Get prediction history
        
        Returns:
            pd.DataFrame: Prediction history records
        """
        if not self.prediction_history:
            return pd.DataFrame()
        
        history_df = pd.DataFrame(self.prediction_history)
        return history_df
    
    def get_evaluation_history(self):
        """
        Get evaluation history
        
        Returns:
            pd.DataFrame: Evaluation history records
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        history_df = pd.DataFrame(self.evaluation_history)
        return history_df


class RiskAnalyzer:
    """
    Risk Analyzer Class
    Used to analyze risk levels and confidence intervals of prediction results
    """
    
    def __init__(self):
        """Initialize risk analyzer"""
        self.risk_thresholds = {
            'low': 2000,
            'medium': 4000,
            'high': 6000
        }
    
    def classify_risk_level(self, prediction):
        """
        Classify risk level based on prediction value
        
        Parameters:
            prediction (float): Predicted medical cost
            
        Returns:
            str: Risk level ('low', 'medium', 'high')
        """
        if prediction <= self.risk_thresholds['low']:
            return 'low'
        elif prediction <= self.risk_thresholds['medium']:
            return 'medium'
        elif prediction <= self.risk_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def calculate_confidence_interval(self, predictions, confidence=0.95):
        """
        Calculate prediction confidence interval
        
        Parameters:
            predictions: Array of prediction values
            confidence (float): Confidence level
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        return lower_bound, upper_bound
    
    def analyze_prediction_risk(self, predictions, actual_values=None):
        """
        Prediction risk analysis
        
        Parameters:
            predictions: Prediction values
            actual_values: Actual values (optional)
            
        Returns:
            dict: Risk analysis results
        """
        risk_analysis = {}
        
        # Risk level distribution
        risk_levels = [self.classify_risk_level(pred) for pred in predictions]
        risk_distribution = pd.Series(risk_levels).value_counts()
        
        risk_analysis['risk_distribution'] = risk_distribution.to_dict()
        
        # Confidence interval
        lower_bound, upper_bound = self.calculate_confidence_interval(predictions)
        risk_analysis['confidence_interval'] = (lower_bound, upper_bound)
        risk_analysis['confidence_level'] = 0.95
        
        # Prediction uncertainty
        prediction_std = np.std(predictions)
        risk_analysis['prediction_uncertainty'] = prediction_std
        
        # If actual values are provided, calculate prediction accuracy
        if actual_values is not None:
            accuracy_analysis = {}
            for risk in ['low', 'medium', 'high', 'very_high']:
                mask = np.array(risk_levels) == risk
                if mask.sum() > 0:
                    risk_actual = actual_values[mask]
                    risk_pred = np.array(predictions)[mask]
                    
                    mae = mean_absolute_error(risk_actual, risk_pred)
                    mape = np.mean(np.abs((risk_actual - risk_pred) / risk_actual)) * 100
                    
                    accuracy_analysis[risk] = {
                        'mae': mae,
                        'mape': mape,
                        'sample_count': mask.sum()
                    }
            
            risk_analysis['accuracy_by_risk'] = accuracy_analysis
        
        return risk_analysis
    
    def generate_risk_recommendations(self, risk_analysis):
        """
        Generate risk recommendations
        
        Parameters:
            risk_analysis (dict): Risk analysis results
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Recommendations based on risk distribution
        risk_dist = risk_analysis['risk_distribution']
        
        if risk_dist.get('high', 0) + risk_dist.get('very_high', 0) > 0.3:
            recommendations.append("• High risk population proportion is relatively high, recommend strengthening preventive health measures")
            recommendations.append("• Consider increasing health screening frequency")
        
        if risk_dist.get('low', 0) < 0.4:
            recommendations.append("• Low risk population proportion is relatively low, may indicate insufficient health awareness")
            recommendations.append("• Recommend strengthening health education and preventive promotion")
        
        # Recommendations based on uncertainty
        uncertainty = risk_analysis['prediction_uncertainty']
        if uncertainty > 1000:
            recommendations.append("• Prediction uncertainty is relatively high, recommend collecting more data to improve the model")
            recommendations.append("• Consider introducing more individual features")
        
        # Recommendations based on confidence interval
        ci_lower, ci_upper = risk_analysis['confidence_interval']
        ci_width = ci_upper - ci_lower
        
        if ci_width > 3000:
            recommendations.append("• Prediction confidence interval is relatively wide, recommend further subdividing risk categories")
            recommendations.append("• Consider developing specialized risk assessment tools")
        
        return recommendations