# -*- coding: utf-8 -*-
"""
数据加载模块
提供从CSV文件加载数据的功能
"""

import pandas as pd
import os


class DataLoader:
    """
    数据加载器类
    负责从CSV文件加载数据
    """
    
    def __init__(self, data_dir='.'):
        """
        初始化数据加载器
        
        Args:
            data_dir (str): 数据目录路径，默认为当前目录
        """
        self.data_dir = data_dir
    
    def load_from_csv(self, filename):
        """
        从CSV文件加载数据
        
        Args:
            filename (str): CSV文件名
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 使用pandas读取CSV文件
        data = pd.read_csv(file_path)
        
        # 清理列名中的空格
        data.columns = data.columns.str.strip()
        
        return data
    
    def get_data_info(self, data):
        """
        获取数据基本信息
        
        Args:
            data (pd.DataFrame): 数据框
            
        Returns:
            dict: 数据信息
        """
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'null_counts': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        return info