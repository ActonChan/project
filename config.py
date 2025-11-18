# -*- coding: utf-8 -*-
"""
Configuration File
Management of hyperparameters for linear regression and gradient boosting models
Supports flexible parameter adjustment and configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class DataConfig:
    """
    Data Configuration Class
    Manages data paths and preprocessing parameters
    """
    # Data file path
    data_path: str = "/Users/kwunchungchan/Library/CloudStorage/坚果云-a524615236@163.com/CUHKSZ/数据分析5918/project/medical_insurance.csv"
    
    # Data split parameters
    test_size: float = 0.2          # Test set ratio
    train_size: float = 0.8         # Train set ratio
    random_state: int = 42          # Random seed
    shuffle_data: bool = True       # Whether to shuffle data
    
    # Preprocessing parameters
    handle_missing_values: str = 'mean'  # Missing values handling: 'mean', 'median', 'drop'
    outlier_method: str = 'iqr'          # Outlier detection method: 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 1.5       # Outlier threshold (IQR method)
    standardize_features: bool = True    # Whether to standardize features
    normalize_target: bool = False       # Whether to normalize target variable


@dataclass
class FeatureConfig:
    """
    Feature Engineering Configuration Class
    Manages feature selection and engineering parameters
    """
    # Categorical feature list
    categorical_features: List[str] = field(default_factory=lambda: [
        'sex', 'smoker', 'region'
    ])
    
    # Numerical feature list
    numerical_features: List[str] = field(default_factory=lambda: [
        'age', 'bmi', 'income', 'visits_last_year', 'hospitalizations_last_3yrs', 
        'days_hospitalized_last_3yrs', 'medication_count', 'systolic_bp', 
        'diastolic_bp', 'ldl', 'hba1c', 'chronic_count'
    ])
    
    # Feature selection parameters
    feature_selection_method: str = 'rfe'     # Feature selection method: 'rfe', 'chi2', 'mutual_info', 'variance_threshold'
    n_features_to_select: int = 10            # Number of features to select (0 means select all)
    feature_selection_threshold: float = 0.01  # Feature selection threshold
    
    # Encoding method
    encoding_method: str = 'label'            # Encoding method: 'label', 'onehot', 'target'
    
    # Feature engineering
    create_interactions: bool = True          # Whether to create interaction features
    create_polynomial_features: bool = False  # Whether to create polynomial features
    polynomial_degree: int = 2               # Polynomial degree


@dataclass
class ModelConfig:
    """
    Model Configuration Class
    Manages hyperparameters for linear regression and gradient boosting algorithms
    """
    # Default model types
    linear_model_types: List[str] = field(default_factory=lambda: ['linear', 'ridge', 'lasso', 'elastic_net'])
    primary_model: str = 'gradient_boosting'  # Primary model: 'linear', 'ridge', 'lasso', 'elastic_net', 'gradient_boosting'
    
    # Cross-validation parameters
    cv_folds: int = 5                      # Number of cross-validation folds
    cv_scoring: str = 'neg_mean_squared_error'  # Cross-validation scoring metric
    
    # Linear Regression parameters
    lr_fit_intercept: bool = True
    lr_normalize: bool = False
    lr_n_jobs: Optional[int] = None
    
    # Ridge Regression parameters
    ridge_alpha: float = 1.0
    ridge_fit_intercept: bool = True
    ridge_random_state: int = 42
    
    # Lasso Regression parameters
    lasso_alpha: float = 1.0
    lasso_fit_intercept: bool = True
    lasso_random_state: int = 42
    lasso_max_iter: int = 2000
    
    # Elastic Net parameters
    elastic_net_alpha: float = 1.0
    elastic_net_l1_ratio: float = 0.5
    elastic_net_fit_intercept: bool = True
    elastic_net_random_state: int = 42
    elastic_net_max_iter: int = 2000
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3
    gb_min_samples_split: int = 2
    gb_min_samples_leaf: int = 1
    gb_subsample: float = 1.0
    gb_max_features: Optional[str] = None
    gb_random_state: int = 42
    gb_validation_fraction: float = 0.1
    gb_n_iter_no_change: int = 10
    gb_tol: float = 1e-4
    
    # Regularization parameters
    regularization_alpha: float = 1.0     # Regularization strength
    l1_ratio: float = 0.5                 # L1/L2 ratio for Elastic Net


@dataclass
class TrainingConfig:
    """
    Training Configuration Class
    Manages model training parameters
    """
    # Training parameters
    batch_size: int = 32             # Batch size
    epochs: int = 100               # Training epochs
    learning_rate: float = 0.001    # Learning rate
    optimizer: str = 'adam'         # Optimizer: 'adam', 'sgd', 'rmsprop'
    
    # Regularization parameters
    dropout_rate: float = 0.0       # Dropout rate
    l1_reg: float = 0.0             # L1 regularization coefficient
    l2_reg: float = 0.0             # L2 regularization coefficient
    
    # Early stopping parameters
    early_stopping: bool = True      # Whether to use early stopping
    patience: int = 10              # Early stopping patience
    min_delta: float = 1e-4         # Early stopping minimum improvement threshold
    
    # Learning rate scheduling
    use_scheduler: bool = False     # Whether to use learning rate scheduler
    scheduler_factor: float = 0.5   # Learning rate scheduling factor
    scheduler_patience: int = 5     # Learning rate scheduling patience


@dataclass
class EvaluationConfig:
    """
    Evaluation Configuration Class
    Manages model evaluation parameters
    """
    # Evaluation metrics
    primary_metric: str = 'rmse'           # Primary evaluation metric: 'rmse', 'mae', 'r2', 'mape'
    metrics: List[str] = field(default_factory=lambda: [
        'rmse', 'mae', 'r2', 'mape', 'explained_variance'
    ])
    
    # Visualization parameters
    create_plots: bool = True              # Whether to create visualization plots
    plot_format: str = 'png'               # Plot format
    plot_dpi: int = 300                    # Plot resolution
    
    # Report generation
    generate_report: bool = True           # Whether to generate evaluation report
    save_predictions: bool = True          # Whether to save prediction results
    
    # Confidence interval
    confidence_level: float = 0.95         # Confidence level
    calculate_uncertainty: bool = True     # Whether to calculate prediction uncertainty


@dataclass
class OutputConfig:
    """
    Output Configuration Class
    Manages result saving and output parameters
    """
    # Output directory
    output_dir: str = "./output/"
    
    # Model-specific result directories
    linear_result_dir: str = "./output/linear_results/"
    gradient_result_dir: str = "./output/gradient_results/"
    
    # File naming
    model_filename: str = "trained_model.pkl"
    predictions_filename: str = "predictions.csv"
    evaluation_report_filename: str = "evaluation_report.txt"
    feature_importance_filename: str = "feature_importance.png"
    comparison_plot_filename: str = "model_comparison.png"
    
    # Save format
    save_format: str = "pickle"            # Model save format: 'pickle', 'joblib'
    
    # Log level
    log_level: str = "INFO"                # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Version control
    version_output: bool = True            # Whether to include version information


class ConfigManager:
    """
    Configuration Manager Class
    Unified management of all configuration classes
    Provides configuration loading, saving, updating, and other functionalities
    """
    
    def __init__(self):
        """Initialize configuration manager"""
        self.data_config = DataConfig()
        self.feature_config = FeatureConfig()
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.evaluation_config = EvaluationConfig()
        self.output_config = OutputConfig()
        
        # Create output directories
        os.makedirs(self.output_config.output_dir, exist_ok=True)
        os.makedirs(self.output_config.linear_result_dir, exist_ok=True)
        os.makedirs(self.output_config.gradient_result_dir, exist_ok=True)
    
    def get_config(self, config_type: str) -> Any:
        """
        Get specified type configuration
        
        Args:
            config_type: Configuration type ('data', 'feature', 'model', 'training', 'evaluation', 'output')
            
        Returns:
            Corresponding configuration object
            
        Raises:
            ValueError: When configuration type does not exist
        """
        config_map = {
            'data': self.data_config,
            'feature': self.feature_config,
            'model': self.model_config,
            'training': self.training_config,
            'evaluation': self.evaluation_config,
            'output': self.output_config
        }
        
        if config_type not in config_map:
            raise ValueError(f"Unknown configuration type: {config_type}. "
                           f"Available types: {list(config_map.keys())}")
        
        return config_map[config_type]
    
    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all configurations
        
        Returns:
            dict: Dictionary containing all configurations
        """
        return {
            'data': self.data_config,
            'feature': self.feature_config,
            'model': self.model_config,
            'training': self.training_config,
            'evaluation': self.evaluation_config,
            'output': self.output_config
        }
    
    def update_config(self, config_type: str, **kwargs) -> None:
        """
        Update specified type configuration
        
        Args:
            config_type: Configuration type
            **kwargs: Configuration parameters dictionary
        """
        config = self.get_config(config_type)
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"Updated {config_type}_config.{key} = {value}")
            else:
                print(f"Warning: {config_type}_config.{key} does not exist, skipping...")
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get hyperparameters for specified model
        
        Args:
            model_type (str): Model type
            
        Returns:
            dict: Model hyperparameters dictionary
        """
        if model_type == 'gradient_boosting':
            return {
                'n_estimators': self.model_config.gb_n_estimators,
                'learning_rate': self.model_config.gb_learning_rate,
                'max_depth': self.model_config.gb_max_depth,
                'subsample': self.model_config.gb_subsample,
                'random_state': self.model_config.gb_random_state
            }
        elif model_type == 'linear_regression':
            return {
                'fit_intercept': self.model_config.lr_fit_intercept,
                'normalize': self.model_config.lr_normalize,
                'n_jobs': self.model_config.lr_n_jobs
            }
        elif model_type == 'ridge':
            return {
                'alpha': self.model_config.ridge_alpha,
                'fit_intercept': self.model_config.ridge_fit_intercept,
                'random_state': self.model_config.ridge_random_state
            }
        elif model_type == 'lasso':
            return {
                'alpha': self.model_config.lasso_alpha,
                'fit_intercept': self.model_config.lasso_fit_intercept,
                'random_state': self.model_config.lasso_random_state,
                'max_iter': self.model_config.lasso_max_iter
            }
        elif model_type == 'elastic_net':
            return {
                'alpha': self.model_config.elastic_net_alpha,
                'l1_ratio': self.model_config.elastic_net_l1_ratio,
                'fit_intercept': self.model_config.elastic_net_fit_intercept,
                'random_state': self.model_config.elastic_net_random_state,
                'max_iter': self.model_config.elastic_net_max_iter
            }
        else:
            return {}
    
    def save_config(self, filepath: str) -> None:
        """
        Save configuration to file
        
        Args:
            filepath: Save path
        """
        import json
        
        config_dict = {
            'data': self.data_config.__dict__,
            'feature': self.feature_config.__dict__,
            'model': self.model_config.__dict__,
            'training': self.training_config.__dict__,
            'evaluation': self.evaluation_config.__dict__,
            'output': self.output_config.__dict__
        }
        
        # Handle non-serializable objects
        def convert_values(obj):
            if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        # Recursively convert all values
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_values(data)
        
        config_dict = recursive_convert(config_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"Configuration saved to: {filepath}")
    
    def load_config(self, filepath: str) -> None:
        """
        Load configuration from file
        
        Args:
            filepath: File path
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Update configurations
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.data_config, key):
                    setattr(self.data_config, key, value)
        
        if 'feature' in config_dict:
            for key, value in config_dict['feature'].items():
                if hasattr(self.feature_config, key):
                    setattr(self.feature_config, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        
        if 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                if hasattr(self.evaluation_config, key):
                    setattr(self.evaluation_config, key, value)
        
        if 'output' in config_dict:
            for key, value in config_dict['output'].items():
                if hasattr(self.output_config, key):
                    setattr(self.output_config, key, value)
        
        print(f"Configuration loaded from: {filepath}")
    
    def reset_to_default(self) -> None:
        """Reset all configurations to default values"""
        self.__init__()
        print("All configurations reset to default values")
    
    def print_config(self) -> None:
        """Print current all configurations"""
        configs = self.get_all_configs()
        
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        
        for category, config in configs.items():
            print(f"\n{category.upper()} CONFIG:")
            print("-" * 30)
            
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    print(f"{key:25}: {value}")
        
        print("=" * 60)


# Global configuration instance
config_manager = ConfigManager()

# Configuration management functions
def get_config(config_type: str) -> Any:
    """
    Get configuration by type
    
    Args:
        config_type: Configuration type
        
    Returns:
        Configuration object
    """
    return config_manager.get_config(config_type)


def update_config(config_type: str, **kwargs) -> None:
    """
    Update configuration
    
    Args:
        config_type: Configuration type
        **kwargs: Configuration parameters
    """
    config_manager.update_config(config_type, **kwargs)


def save_config_to_file(filepath: str) -> None:
    """
    Save configuration to file
    
    Args:
        filepath: Save path
    """
    config_manager.save_config(filepath)


def load_config_from_file(filepath: str) -> None:
    """
    Load configuration from file
    
    Args:
        filepath: File path
    """
    config_manager.load_config(filepath)


def print_current_config() -> None:
    """Print current configuration"""
    config_manager.print_config()


# Predefined configuration templates
PRESET_CONFIGS = {
    'fast_training': {
        'model': {'gb_n_estimators': 50},
        'training': {'early_stopping': False},
        'evaluation': {'create_plots': False}
    },
    'high_accuracy': {
        'model': {'gb_n_estimators': 200},
        'training': {'early_stopping': True},
        'evaluation': {'create_plots': True}
    },
    'cross_validation': {
        'model': {'cv_folds': 10},
        'training': {'early_stopping': True},
        'evaluation': {'generate_report': True, 'save_predictions': True}
    }
}


def apply_preset_config(preset_name: str):
    """
    Apply predefined configuration template
    
    Parameters:
        preset_name (str): Template name ('fast_training', 'high_accuracy', 'cross_validation')
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown configuration template: {preset_name}")
    
    preset = PRESET_CONFIGS[preset_name]
    
    for category, params in preset.items():
        config_manager.update_config(category, **params)
    
    print(f"Applied configuration template: {preset_name}")
    config_manager.print_current_config()


if __name__ == "__main__":
    # Test configuration manager
    print("Testing configuration manager")
    config_manager.print_current_config()
    
    # Test update configuration
    print("\nTesting configuration update")
    config_manager.update_config('model', gb_n_estimators=200)
    config_manager.print_current_config()
    
    # Test get model parameters
    print("\nTesting getting model parameters")
    gb_params = config_manager.get_model_params('gradient_boosting')
    print(gb_params)
    
    # Test apply preset configuration
    print("\nTesting applying fast training configuration")
    apply_preset_config('fast_training')