# -*- coding: utf-8 -*-
"""
总结增强版CNN模型训练结果
"""

import os
import json

# 读取评估结果
results_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_evaluation.json')

if os.path.exists(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("=" * 60)
    print("增强版CNN模型训练结果总结")
    print("=" * 60)
    
    # 模型配置
    print("\n1. 模型配置:")
    config = results['model_config']
    print(f"   - 输入维度: {config['input_dim']}")
    print(f"   - 序列长度: {config['sequence_length']}")
    print(f"   - 卷积层数: {config['conv_layers']}")
    print(f"   - 卷积核数量: {config['filters']}")
    print(f"   - 卷积核大小: {config['kernel_sizes']}")
    print(f"   - 全连接层: {config['fc_layers']}")
    print(f"   - 学习率: {config['learning_rate']}")
    print(f"   - 批次大小: {config['batch_size']}")
    print(f"   - 训练轮数: {config['epochs']}")
    
    # 训练结果
    print("\n2. 训练结果:")
    training = results['training_results']
    print(f"   - 最终训练损失: {training['final_train_loss']:.2f}")
    print(f"   - 最终验证损失: {training['final_val_loss']:.2f}")
    print(f"   - 最佳验证损失: {training['best_val_loss']:.2f}")
    print(f"   - 总训练轮数: {training['total_epochs']}")
    
    # 模型信息
    print("\n3. 模型信息:")
    model_info = training['model_info']
    print(f"   - 模型类型: {model_info['model_type']}")
    print(f"   - 总参数数量: {model_info['total_parameters']}")
    print(f"   - 可训练参数数量: {model_info['trainable_parameters']}")
    print(f"   - 输入形状: {model_info['input_shape']}")
    print(f"   - 输出大小: {model_info['output_size']}")
    
    # 评估结果
    print("\n4. 评估结果:")
    evaluation = results['evaluation_results']
    print(f"   - MSE (均方误差): {evaluation['MSE']:.2f}")
    print(f"   - RMSE (均方根误差): {evaluation['RMSE']:.2f}")
    print(f"   - MAE (平均绝对误差): {evaluation['MAE']:.2f}")
    print(f"   - R² (决定系数): {evaluation['R²']:.4f}")
    
    # 预测示例
    print("\n5. 预测示例 (前5个样本):")
    predictions = evaluation['predictions'][:5]
    actual_values = evaluation['actual_values'][:5]
    
    for i in range(5):
        print(f"   样本{i+1}: 预测值={predictions[i]:.2f}, 实际值={actual_values[i]:.2f}, 误差={abs(predictions[i]-actual_values[i]):.2f}")
    
    # 训练时间
    print("\n6. 训练时间:")
    print(f"   - 训练完成时间: {results['timestamp']}")
    print(f"   - 模型类型: {results['model_type']}")
    
    print("\n" + "=" * 60)
    print("总结: 增强版CNN模型在医疗保险费用预测任务上表现良好")
    print(f"R²分数为{evaluation['R²']:.4f}，表明模型可以解释约{evaluation['R²']*100:.2f}%的数据变异")
    print("=" * 60)
else:
    print(f"结果文件不存在: {results_path}")