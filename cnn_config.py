# -*- coding: utf-8 -*-
"""
CNN模型配置类
管理模型的超参数和网络架构设置
"""

from typing import List


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