# -*- coding: utf-8 -*-
"""
CNN模型主程序入口
整合所有模块，提供完整的模型训练和评估流程
"""

from datetime import datetime
import json
import os
from sklearn.model_selection import train_test_split

# 导入自定义模块
from cnn_config import CNNConfig
from data_processor import InsuranceDataProcessor
from cnn_model_definition import CNNModel
from cnn_trainer import CNNTrainer


def run_cnn_example(): 
    """
    运行CNN模型示例
    完整的数据加载、预处理、模型训练和评估流程
    """
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
        import numpy as np
        import torch
        
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