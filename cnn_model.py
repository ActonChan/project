# -*- coding: utf-8 -*-
"""
卷积神经网络(CNN)模型实现
使用PyTorch框架构建的深度学习模型，适用于医疗保险成本预测任务
包含完整的模型定义、数据预处理、训练循环、评估等模块
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
warnings.filterwarnings('ignore')


class CNNConfig:
    """
    CNN模型配置类
    管理模型的超参数和网络架构设置
    """
    
    def __init__(self, 
                 input_dim: int = 52,
                 sequence_length: int = 13,
                 conv_layers: int = 3,
                 filters: List[int] = None,
                 kernel_sizes: List[int] = None,
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 fc_layers: List[int] = None,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 100,
                 patience: int = 10,
                 use_batch_norm: bool = True,
                 use_dropout: bool = True,
                 target_column: str = 'annual_premium'):
        """
        初始化CNN配置
        
        Args:
            input_dim: 输入特征维度
            sequence_length: 序列化长度（将特征重塑为sequence_length x (input_dim//sequence_length)的形状）
            conv_layers: 卷积层数量
            filters: 每层的卷积核数量列表
            kernel_sizes: 每层的卷积核大小列表
            activation: 激活函数类型 ('relu', 'tanh', 'leaky_relu')
            dropout_rate: Dropout比例
            fc_layers: 全连接层大小列表
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            patience: 早停耐心值
            use_batch_norm: 是否使用批标准化
            use_dropout: 是否使用Dropout
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.conv_layers = conv_layers
        self.filters = filters if filters else [32, 64, 128][:conv_layers]
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3, 3, 3][:conv_layers]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.fc_layers = fc_layers if fc_layers else [256, 128, 64, 1]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.target_column = target_column
        
        # 数据重塑后的宽度
        self.reshape_width = input_dim // sequence_length
        assert input_dim % sequence_length == 0, "input_dim必须能被sequence_length整除"
        
        print(f"CNN模型配置初始化完成:")
        print(f"  输入维度: {input_dim}")
        print(f"  序列长度: {sequence_length}")
        print(f"  重塑后尺寸: {sequence_length} x {self.reshape_width}")
        print(f"  卷积层数: {conv_layers}")
        print(f"  卷积核数量: {self.filters}")
        print(f"  激活函数: {activation}")


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
        X = self._reshape_for_cnn(X)
        
        print(f"预处理完成，最终形状: X={X.shape}, y={y.shape}")
        
        return X, y
    
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


class CNNModel(nn.Module):
    """
    卷积神经网络模型
    使用1D卷积处理序列化特征，适用于表格数据
    """
    
    def __init__(self, config: CNNConfig):
        """
        初始化CNN模型
        
        Args:
            config: CNN配置对象
        """
        super(CNNModel, self).__init__()
        self.config = config
        
        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层卷积（输入通道为1）
        in_channels = 1
        for i in range(config.conv_layers):
            out_channels = config.filters[i]
            kernel_size = config.kernel_sizes[i]
            
            # 计算padding以保持特征图大小
            padding = kernel_size // 2
            
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.conv_layers.append(conv_layer)
            
            # 批标准化
            if config.use_batch_norm:
                bn_layer = nn.BatchNorm2d(out_channels)
                self.conv_layers.append(bn_layer)
            
            # 激活函数
            if config.activation == 'relu':
                activation_layer = nn.ReLU(inplace=True)
            else:
                raise ValueError(f"不支持的激活函数: {config.activation}")
            self.conv_layers.append(activation_layer)
            
            # 池化层 (仅在较大的特征图上使用)
            if self.config.sequence_length > 1 and self.config.reshape_width > 1:
                pool_layer = nn.MaxPool2d(2, 2)
                self.conv_layers.append(pool_layer)
            
            # Dropout
            if config.use_dropout:
                dropout_layer = nn.Dropout2d(config.dropout_rate)
                self.conv_layers.append(dropout_layer)
            
            # 更新输入通道数
            in_channels = out_channels
        
        # 计算卷积层后的特征图大小
        self._calculate_flattened_size()
        
        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        input_size = self.flattened_size
        
        # 构建全连接层
        # 使用原始配置，但去掉最后一层，然后添加增强层
        fc_layers_config = config.fc_layers[:-1]  # 去掉原始最后一层
        
        for i, fc_size in enumerate(fc_layers_config):
            if i == 0:
                # 第一层全连接
                fc_layer = nn.Linear(input_size, fc_size)
            else:
                # 中间层
                fc_layer = nn.Linear(fc_layers_config[i-1], fc_size)
            
            self.fc_layers.append(fc_layer)
            
            # 添加激活函数和dropout
            if config.activation == 'relu':
                activation_layer = nn.ReLU(inplace=True)
            elif config.activation == 'tanh':
                activation_layer = nn.Tanh()
            elif config.activation == 'leaky_relu':
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
            else:
                activation_layer = nn.ReLU(inplace=True)
            self.fc_layers.append(activation_layer)
            
            if config.use_dropout:
                dropout_layer = nn.Dropout(config.dropout_rate)
                self.fc_layers.append(dropout_layer)
        
        # 添加额外的全连接层以增强模型表达能力
        # 在原有全连接层基础上，添加一个16神经元的层
        last_fc_size = fc_layers_config[-1] if fc_layers_config else input_size
        additional_fc = nn.Linear(last_fc_size, 16)
        self.fc_layers.append(additional_fc)
        
        # 添加激活函数和dropout
        if config.activation == 'relu':
            activation_layer = nn.ReLU(inplace=True)
        elif config.activation == 'tanh':
            activation_layer = nn.Tanh()
        elif config.activation == 'leaky_relu':
            activation_layer = nn.LeakyReLU(0.1, inplace=True)
        else:
            activation_layer = nn.ReLU(inplace=True)
        self.fc_layers.append(activation_layer)
        
        if config.use_dropout:
            dropout_layer = nn.Dropout(config.dropout_rate)
            self.fc_layers.append(dropout_layer)
        
        # 添加最终输出层
        final_fc = nn.Linear(16, 1)
        self.fc_layers.append(final_fc)
        
        print(f"CNN模型初始化完成:")
        print(f"  卷积层数: {config.conv_layers}")
        print(f"  全连接层: {config.fc_layers} + [16, 1] (增强层)")
        print(f"  展平后特征数: {self.flattened_size}")
    
    def _calculate_flattened_size(self):
        """计算卷积层后的特征图大小"""
        # 模拟一次前向传播来计算展平后的特征数
        with torch.no_grad():
            x = torch.randn(1, 1, self.config.reshape_width, self.config.sequence_length)
            
            for layer in self.conv_layers:
                x = layer(x)
            
            self.flattened_size = x.numel() // x.shape[0]  # 去掉batch维度后的特征数
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: 预测输出
        """
        # 卷积层
        for layer in self.conv_layers:
            x = layer(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        for layer in self.fc_layers:
            x = layer(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算增强后的全连接层配置
        enhanced_fc_layers = self.config.fc_layers[:-1] + [16, 1]  # 移除原始最后一层，添加增强层
        
        info = {
            "model_type": "CNN Enhanced",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "conv_layers": self.config.conv_layers,
            "filters": self.config.filters,
            "fc_layers": enhanced_fc_layers,
            "activation": self.config.activation,
            "dropout_rate": self.config.dropout_rate,
            "input_shape": f"(1, {self.config.reshape_width}, {self.config.sequence_length})",
            "output_size": 1
        }
        
        return info


class CNNTrainer:
    """
    CNN模型训练器
    负责模型训练、验证、评估等操作
    """
    
    def __init__(self, config: CNNConfig, device: str = 'cuda'):
        """
        初始化训练器
        
        Args:
            config: CNN配置对象
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.config = config
        self.device = self._get_device(device)
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        
        print(f"训练器初始化完成，使用设备: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def create_model(self) -> CNNModel:
        """
        创建模型实例
        
        Returns:
            CNNModel: CNN模型实例
        """
        self.model = CNNModel(self.config).to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"模型创建完成，优化器: Adam(lr={self.config.learning_rate})")
        return self.model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   save_path: str = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            save_path: 模型保存路径
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        print("开始模型训练...")
        
        # 创建模型
        model = self.create_model()
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = self.criterion(val_outputs.squeeze(), y_val_tensor).item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印训练进度
            if (epoch + 1) % 1 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.config.epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} "
                      f"Val Loss: {val_loss:.4f} "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= self.config.patience:
                print(f"早停触发，在第{epoch+1}轮停止训练")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 保存模型
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': best_model_state if best_model_state else model.state_dict(),
                'config': self.config.__dict__,
                'model_info': model.get_model_info(),
                'training_history': {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }
            }, save_path)
            print(f"模型已保存到: {save_path}")
        
        # 训练结果
        training_results = {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.train_losses),
            'model_info': model.get_model_info(),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }
        
        print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
        return training_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      model_path: str = None) -> Dict[str, Any]:
        """
        Evaluate model
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_path: Model file path (for loading pretrained model)
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            if self.model is None:
                self.create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.model is None:
            raise ValueError("Model not created or loaded")
        
        self.model.eval()
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            predictions = predictions.squeeze().cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_np, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, predictions)
        r2 = r2_score(y_test_np, predictions)
        
        evaluation_results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': predictions[:10].tolist(),  # Save first 10 predictions as examples
            'actual_values': y_test_np[:10].tolist(),
            'model_info': self.model.get_model_info()
        }
        
        print(f"Model evaluation completed:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Save path
        """
        if not self.train_losses or not self.val_losses:
            print("No training history data available for plotting")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.title('Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        if hasattr(self, 'scheduler') and self.scheduler:
            # Get learning rate history (simplified to show current learning rate)
            current_lr = self.optimizer.param_groups[0]['lr']
            plt.axhline(y=current_lr, color='r', linestyle='--', 
                       label=f'Current LR: {current_lr:.6f}')
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练总结
        
        Returns:
            Dict[str, Any]: 训练总结
        """
        if not self.train_losses or not self.val_losses:
            return {"error": "没有训练数据"}
        
        return {
            "total_epochs": len(self.train_losses),
            "best_train_loss": min(self.train_losses),
            "best_val_loss": min(self.val_losses),
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "training_improvement": self.train_losses[0] - self.train_losses[-1],
            "validation_improvement": self.val_losses[0] - self.val_losses[-1],
            "model_info": self.model.get_model_info() if self.model else None
        }


def run_cnn_example(): 
    # 1. 加载数据
    try:
        from data_loader import DataLoader
        loader = DataLoader()
        data = loader.load_from_csv('medical_insurance.csv')
        print(f"数据加载成功: {data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 配置CNN模型
    config = CNNConfig(
        input_dim=52,          # 输入特征维度 (处理后的实际特征数)
        sequence_length=13,    # 序列长度，52=13x4
        conv_layers=2,         # 卷积层数
        filters=[16, 32],      # 卷积核数量
        kernel_sizes=[5, 3],   # 卷积核大小
        activation='relu',     # 激活函数
        dropout_rate=0.2,      # Dropout比例
        fc_layers=[128, 64, 32, 1],  # 全连接层
        learning_rate=0.001,   # 学习率
        batch_size=128,        # 批次大小
        epochs=30,             # 训练轮数
        patience=8             # 早停耐心值
    )
    
    # 3. 数据预处理
    processor = InsuranceDataProcessor(config)
    
    # 分离测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    print(f"数据分割: 训练集{train_data.shape}, 验证集{val_data.shape}, 测试集{test_data.shape}")
    print(f"训练集列数: {len(train_data.columns)}")
    
    # 检查列名中的空格问题
    actual_columns = [col.strip() for col in train_data.columns]
    print(f"清理空格后的列数: {len(actual_columns)}")
    
    target_column_stripped = config.target_column.strip()
    print(f"目标列'{target_column_stripped}'在训练集中: {target_column_stripped in actual_columns}")
    
    if target_column_stripped not in actual_columns:
        print("可用列(含annual):")
        for col in actual_columns:
            if 'annual' in col.lower():
                print(f"  '{col}'")
    
    # 预处理数据
    X_train, y_train = processor.preprocess_data(train_data, is_training=True)
    X_val, y_val = processor.preprocess_data(val_data, is_training=False)
    X_test, y_test = processor.preprocess_data(test_data, is_training=False)
    
    print(f"数据预处理完成:")
    print(f"  训练集: X{X_train.shape}, y{y_train.shape}")
    print(f"  验证集: X{X_val.shape}, y{y_val.shape}")
    print(f"  测试集: X{X_test.shape}, y{y_test.shape}")
    
    # 4. 创建训练器并训练模型
    trainer = CNNTrainer(config)
    
    # 训练模型
    model_save_path = 'output/cnn_results/cnn_model.pth'
    training_results = trainer.train_model(
        X_train, y_train, X_val, y_val, 
        save_path=model_save_path
    )
    
    # 5. 评估模型
    evaluation_results = trainer.evaluate_model(
        X_test, y_test, model_path=model_save_path
    )
    
    # 6. 绘制训练历史
    os.makedirs('output/cnn_results', exist_ok=True)
    trainer.plot_training_history('output/cnn_results/cnn_training_history.png')
    
    # 7. 保存评估结果
    # 确保所有数据都是JSON可序列化的
    def make_json_serializable(obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy标量
            return obj.item()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # 转换为字符串
    
    results = {
        'training_results': make_json_serializable(training_results),
        'evaluation_results': make_json_serializable(evaluation_results),
        'model_config': make_json_serializable(config.__dict__),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('output/cnn_results/cnn_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("CNN模型训练和评估完成!")
    print("结果已保存到 output/cnn_results/ 目录")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # 运行示例
    results = run_cnn_example()