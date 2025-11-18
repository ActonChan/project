# -*- coding: utf-8 -*-
"""
Data Import Module
Responsible for data reading and preliminary loading functionality
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Union


class DataLoader:
    """
    Data Loader Class
    Responsible for reading and loading datasets from filesystem
    """
    
    def __init__(self):
        """Initialize data loader"""
        self.data = None
        self.file_path = None
        
    def load_from_csv(self, file_path):
        """
        Load data from CSV file
        
        Parameters:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: When file doesn't exist
            pd.errors.EmptyDataError: When file is empty
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file does not exist: {file_path}")
        
        try:
            # Read CSV file
            self.data = pd.read_csv(file_path)
            self.file_path = file_path
            
            print(f"Successfully loaded data file: {file_path}")
            print(f"Dataset size: {self.data.shape}")
            print(f"Column names: {list(self.data.columns)}")
            
            return self.data
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading data file: {e}")
    
    def get_data_info(self):
        """
        Get basic information about the dataset
        
        Returns:
            dict: Dictionary containing dataset basic information
        """
        if self.data is None:
            return {"error": "Data not loaded yet"}
        
        info = {
            "data_shape": self.data.shape,
            "num_columns": len(self.data.columns),
            "num_rows": len(self.data),
            "missing_values": self.data.isnull().sum().sum(),
            "data_types": self.data.dtypes.to_dict()
        }
        
        return info
    
    def preview_data(self, n_rows=5):
        """
        Preview first few rows of dataset
        
        Parameters:
            n_rows (int): Number of rows to preview, default 5 rows
            
        Returns:
            pd.DataFrame: Preview data
        """
        if self.data is None:
            print("Data not loaded")
            return None
        
        print(f"Dataset preview (first {n_rows} rows):")
        return self.data.head(n_rows)
    
    def get_column_info(self):
        """
        Get detailed column information
        
        Returns:
            pd.DataFrame: DataFrame containing column information
        """
        if self.data is None:
            print("Data not loaded")
            return None
        
        column_info = pd.DataFrame({
            'Column': self.data.columns,
            'Type': self.data.dtypes,
            'Non-Null Count': self.data.count(),
            'Null Count': self.data.isnull().sum(),
            'Unique Values': self.data.nunique()
        })
        
        return column_info