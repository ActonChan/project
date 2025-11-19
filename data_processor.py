# -*- coding: utf-8 -*-
"""
数据预处理器
负责将医疗保险数据转换为适合CNN训练的格式
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

from cnn_config import CNNConfig


class InsuranceDataProcessor:
    """
    数据预处理器
    负责将医疗保险数据转换为适合CNN训练的格式
    """
    
    def __init__(self, config: CNNConfig):
        """
        初始化数据处理器
        
        Args:
            config: CNN配置对象
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        # 若配置对象已显式指定目标列名则采用之，否则默认使用'annual_premium'作为目标列
        self.target_column = config.target_column if hasattr(config, 'target_column') else 'annual_premium'
        
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理
        
        Args:
            df: 原始数据DataFrame
            is_training: 是否为训练阶段
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) 预处理后的特征和标签
        """
        print(f"开始数据预处理，原始数据形状: {df.shape}")
        
        # 复制数据避免修改原始数据
        data = df.copy()
        
        # 清理列名中的空格
        data.columns = [col.strip() for col in data.columns]
        
        # 移除ID列和非预测相关列
        id_columns = ['person_id']
        data = data.drop(columns=[col for col in id_columns if col in data.columns])
        
        # 分离特征和目标变量
        # 去除目标列名两端空白，防止因列名前后空格导致查找失败
        target_column_stripped = self.target_column.strip()
        if target_column_stripped not in data.columns:
            raise ValueError(f"目标列 '{target_column_stripped}' 不存在于数据中")
        
        X = data.drop(columns=[target_column_stripped])
        y = data[target_column_stripped].values
        
        print(f"特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        # 分离分类特征和数值特征
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        original_numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"分类特征: {categorical_columns}")
        print(f"原始数值特征: {original_numerical_columns}")
        print(f"分类特征: {len(categorical_columns)}个")
        print(f"原始数值特征: {len(original_numerical_columns)}个")
        
        # 处理分类特征
        for col in categorical_columns:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # 处理未见过的标签
                    unique_values = set(X[col].unique())
                    known_values = set(le.classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        print(f"警告: 特征{col}中发现未见过的值: {unknown_values}")
                        # 用最常见的值替换
                        most_common = le.classes_[0]
                        X[col] = X[col].replace(list(unknown_values), most_common)
                    
                    X[col] = le.transform(X[col].astype(str))
                else:
                    print(f"警告: 训练时未遇到特征{col}，使用默认值")
                    X[col] = 0
        
        # 只处理原始的数值特征，避免重复处理编码后的分类特征
        if original_numerical_columns:
            print(f"标准化数值特征: {len(original_numerical_columns)}个")
            if is_training:
                X[original_numerical_columns] = self.scaler.fit_transform(X[original_numerical_columns])
                self.feature_columns = X.columns.tolist()
            else:
                X[original_numerical_columns] = self.scaler.transform(X[original_numerical_columns])
        else:
            # 如果没有原始数值特征，仍需要设置feature_columns
            if is_training:
                self.feature_columns = X.columns.tolist()
        
        numerical_columns = original_numerical_columns  # 更新变量名以保持代码兼容性
        
        # 将DataFrame转换为numpy数组
        X = X.values
        
        # 将数据重塑为CNN期望的格式
        X_reshaped = self._reshape_for_cnn(X)
        
        print(f"预处理完成，最终形状: X={X_reshaped.shape}, y={y.shape}")
        
        return X_reshaped, y
    
    def _reshape_for_cnn(self, X: np.ndarray) -> np.ndarray:
        """
        将特征数据重塑为CNN期望的2D格式
        
        Args:
            X: 原始特征数据
            
        Returns:
            np.ndarray: 重塑后的数据
        """
        sequence_length = self.config.sequence_length
        feature_width = self.config.reshape_width
        
        print(f"重塑数据: {X.shape} -> ({X.shape[0]}, {sequence_length}, {feature_width})")
        
        # 将数据重塑为 (batch_size, sequence_length, feature_width)
        X_reshaped = X.reshape(X.shape[0], sequence_length, feature_width)
        
        # 转换为PyTorch期望的格式 (batch_size, channels, height, width)
        # 这里我们将channel设为1，height=sequence_length，width=feature_width
        X_reshaped = X_reshaped.transpose(0, 2, 1)  # (batch, width, length)
        X_reshaped = X_reshaped.reshape(X.shape[0], 1, feature_width, sequence_length)
        
        return X_reshaped