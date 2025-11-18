# -*- coding: utf-8 -*-
"""
Data Preprocessing Module
Contains data cleaning, transformation, and feature engineering functionality
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Data Preprocessor Class
    Responsible for data cleaning, feature engineering, and feature selection
    """
    
    def __init__(self):
        """
        Initialize data preprocessor
        """
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = None
        self.selected_features = None
        self.numeric_features = ['age', 'bmi', 'income', 'visits_last_year', 
                               'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs', 
                               'medication_count', 'systolic_bp', 'diastolic_bp', 
                               'ldl', 'hba1c', 'chronic_count']
        self.categorical_features = ['sex', 'smoker', 'region', 'urban_rural']
        self.target = 'annual_premium'
        
    def basic_cleaning(self, df):
        """
        Basic data cleaning: handle missing values and duplicates
        
        Parameters:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        print(f"Dataset size before cleaning: {df.shape}")
        
        # Remove duplicate rows
        df_clean = df.drop_duplicates()
        print(f"After removing duplicates: {df_clean.shape}")
        
        # Handle missing values
        missing_info = df_clean.isnull().sum()
        print(f"\nMissing values statistics:")
        print(missing_info[missing_info > 0])
        
        # Remove columns with too many missing values (>50%)
        missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100)
        cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
        if cols_to_drop:
            print(f"Removing columns with excessive missing values: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Remove rows with missing values
        df_clean = df_clean.dropna()
        print(f"After handling missing values: {df_clean.shape}")
        
        return df_clean
    
    def remove_outliers(self, df, method='iqr', threshold=1.5):
        """
        Remove outliers
        
        Parameters:
            df (pd.DataFrame): Input data
            method (str): Outlier detection method ('iqr' or 'zscore')
            threshold (float): Threshold parameter
            
        Returns:
            pd.DataFrame: Data after outlier removal
        """
        df_no_outliers = df.copy()
        removed_rows = 0
        
        # Detect outliers for numerical features
        for col in self.numeric_features:
            if col in df_no_outliers.columns:
                if method == 'iqr':
                    Q1 = df_no_outliers[col].quantile(0.25)
                    Q3 = df_no_outliers[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Remove outliers
                    mask = (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
                    removed_rows += len(df_no_outliers) - mask.sum()
                    df_no_outliers = df_no_outliers[mask]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df_no_outliers[col]))
                    mask = z_scores < threshold
                    removed_rows += len(df_no_outliers) - mask.sum()
                    df_no_outliers = df_no_outliers[mask]
        
        print(f"Removed {removed_rows} outlier rows using {method} method")
        print(f"Dataset size after outlier handling: {df_no_outliers.shape}")
        
        return df_no_outliers
    
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features
        
        Parameters:
            df (pd.DataFrame): Input data
            is_training (bool): Whether in training phase
            
        Returns:
            pd.DataFrame: Encoded data
        """
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                if is_training:
                    # Training phase: create new encoder
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                else:
                    # Prediction phase: use trained encoder
                    if col in self.label_encoders:
                        # Handle unseen labels
                        le = self.label_encoders[col]
                        unique_values = set(df_encoded[col].unique())
                        known_values = set(le.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            print(f"Warning: Unseen values in feature {col}: {unknown_values}")
                            # Replace unseen values with most common value
                            most_common = le.classes_[0]
                            df_encoded[col] = df_encoded[col].replace(list(unknown_values), most_common)
                        
                        df_encoded[col] = le.transform(df_encoded[col])
        
        return df_encoded
    
    def feature_engineering(self, df, is_training=True):
        """
        Feature engineering: create new features
        
        Parameters:
            df (pd.DataFrame): Input data
            is_training (bool): Whether in training phase
            
        Returns:
            pd.DataFrame: Data after feature engineering
        """
        df_featured = df.copy()
        
        # Age group feature
        if 'age' in df_featured.columns:
            df_featured['age_group'] = pd.cut(df_featured['age'], 
                                            bins=[0, 18, 30, 45, 60, 100], 
                                            labels=[0, 1, 2, 3, 4])
            df_featured['age_group'] = df_featured['age_group'].astype(int)
        
        # BMI group feature
        if 'bmi' in df_featured.columns:
            df_featured['bmi_group'] = pd.cut(df_featured['bmi'], 
                                            bins=[0, 18.5, 25, 30, 100], 
                                            labels=[0, 1, 2, 3])
            df_featured['bmi_group'] = df_featured['bmi_group'].astype(int)
        
        # Risk score feature
        if all(col in df_featured.columns for col in ['age', 'bmi', 'chronic_count', 'hospitalizations_last_3yrs']):
            df_featured['risk_score'] = (df_featured['age'] / 100 + 
                                       df_featured['bmi'] / 50 + 
                                       df_featured['chronic_count'] / 5 + 
                                       df_featured['hospitalizations_last_3yrs'] / 3)
        
        # Health index
        if all(col in df_featured.columns for col in ['systolic_bp', 'diastolic_bp', 'ldl', 'hba1c']):
            # Blood pressure index
            df_featured['bp_index'] = (df_featured['systolic_bp'] + df_featured['diastolic_bp']) / 2
            # Metabolic index
            df_featured['metabolic_index'] = (df_featured['ldl'] + df_featured['hba1c']) / 2
        
        # Medical history features
        if all(col in df_featured.columns for col in ['visits_last_year', 'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs']):
            # Average hospital days
            df_featured['avg_hospital_days'] = (df_featured['days_hospitalized_last_3yrs'] / 
                                              (df_featured['hospitalizations_last_3yrs'] + 1))
            # Medical frequency index
            df_featured['medical_frequency'] = (df_featured['visits_last_year'] + 
                                              df_featured['hospitalizations_last_3yrs'])
        
        return df_featured
    
    def select_features(self, X, y, method='rfe', n_features=None):
        """
        Feature selection
        
        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Feature selection method ('rfe', 'kbest', 'rf_importance')
            n_features (int): Number of features to select
            
        Returns:
            list: Selected feature names list
        """
        if n_features is None:
            n_features = min(15, X.shape[1] // 2)
        
        if method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
        elif method == 'kbest':
            # Statistical test-based feature selection
            selector = SelectKBest(score_func=f_regression, k=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rf_importance':
            # Random Forest feature importance-based selection
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
        
        print(f"Selected {len(selected_features)} features using {method} method")
        return selected_features
    
    def scale_features(self, X_train, X_test=None, scaler_type='standard'):
        """
        Feature scaling
        
        Parameters:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            scaler_type (str): Scaler type ('standard', 'robust')
            
        Returns:
            tuple: Scaled features and scaler
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
            df (pd.DataFrame): Complete dataset
            test_size (float): Test set ratio
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Ensure target column exists
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' does not exist in dataset")
        
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, df, test_size=0.2, random_state=42, 
                          remove_outliers=True, feature_selection=True):
        """
        Complete data preprocessing pipeline
        
        Parameters:
            df (pd.DataFrame): Raw data
            test_size (float): Test set ratio
            random_state (int): Random seed
            remove_outliers (bool): Whether to remove outliers
            feature_selection (bool): Whether to perform feature selection
            
        Returns:
            dict: Dictionary containing preprocessed data
        """
        print("Starting data preprocessing pipeline...")
        
        # 1. Basic cleaning
        df_clean = self.basic_cleaning(df)
        
        # 2. Remove outliers
        if remove_outliers:
            df_clean = self.remove_outliers(df_clean, method='iqr', threshold=1.5)
        
        # 3. Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, is_training=True)
        
        # 4. Feature engineering
        df_featured = self.feature_engineering(df_encoded, is_training=True)
        
        # 5. Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df_featured, test_size, random_state
        )
        
        # 6. Feature selection
        if feature_selection and len(X_train) > 0:
            self.selected_features = self.select_features(X_train, y_train, method='rfe')
            X_train = X_train[self.selected_features]
            X_test = X_test[self.selected_features]
        
        # 7. Feature scaling
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save feature column order
        self.feature_columns = list(X_train_scaled.columns)
        
        results = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_columns,
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        print("Data preprocessing completed!")
        return results