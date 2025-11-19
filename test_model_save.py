# -*- coding: utf-8 -*-
"""
测试模型保存功能
"""

import os
import sys
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# 导入增强后的模型模块
from cnn_model import CNNConfig, InsuranceDataProcessor, CNNTrainer
from data_loader import DataLoader

def test_model_save():
    """测试模型保存功能"""
    print("=" * 60)
    print("测试模型保存功能")
    print("=" * 60)
    
    # 1. 加载数据
    try:
        loader = DataLoader()
        data = loader.load_from_csv('medical_insurance.csv')
        print(f"数据加载成功: {data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 配置增强版CNN模型
    config = CNNConfig(
        input_dim=52,          # 输入特征维度
        sequence_length=13,    # 序列长度
        conv_layers=2,         # 卷积层数
        filters=[16, 32],      # 卷积核数量
        kernel_sizes=[5, 3],   # 卷积核大小
        activation='relu',     # 激活函数
        dropout_rate=0.2,      # Dropout比例
        fc_layers=[128, 64, 32, 1],  # 全连接层（模型会自动添加16神经元的层）
        learning_rate=0.001,   # 学习率
        batch_size=256,        # 批次大小
        epochs=2,              # 训练轮数 - 减少到2轮用于测试
        patience=5,            # 早停耐心值
        target_column='annual_premium'  # 目标列名
    )
    
    # 3. 数据预处理
    processor = InsuranceDataProcessor(config)
    
    # 分离测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    print(f"数据分割: 训练集{train_data.shape}, 验证集{val_data.shape}, 测试集{test_data.shape}")
    
    # 预处理数据
    X_train, y_train = processor.preprocess_data(train_data, is_training=True)
    X_val, y_val = processor.preprocess_data(val_data, is_training=False)
    X_test, y_test = processor.preprocess_data(test_data, is_training=False)
    
    print(f"数据预处理完成:")
    print(f"  训练集: X{X_train.shape}, y{y_train.shape}")
    print(f"  验证集: X{X_val.shape}, y{y_val.shape}")
    print(f"  测试集: X{X_test.shape}, y{y_test.shape}")
    
    # 4. 创建训练器并训练模型
    trainer = CNNTrainer(config, device='cuda')  # 使用CUDA进行训练
    
    # 训练模型
    model_save_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'test_enhanced_cnn_model.pth')
    print(f"模型将保存到: {model_save_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    print(f"目录已创建或已存在: {os.path.dirname(model_save_path)}")
    
    training_results = trainer.train_model(
        X_train, y_train, X_val, y_val, 
        save_path=model_save_path
    )
    
    # 检查模型是否保存成功
    if os.path.exists(model_save_path):
        print(f"模型保存成功: {model_save_path}")
        print(f"文件大小: {os.path.getsize(model_save_path)} 字节")
    else:
        print(f"模型保存失败: {model_save_path}")
    
    return training_results


if __name__ == "__main__":
    # 运行测试
    results = test_model_save()