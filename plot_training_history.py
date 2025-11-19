# -*- coding: utf-8 -*-
"""
绘制训练历史
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取评估结果
results_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_evaluation.json')

if os.path.exists(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 提取训练历史
    train_losses = results['training_results']['training_history']['train_losses']
    val_losses = results['training_results']['training_history']['val_losses']
    
    # 绘制训练历史
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 评估结果
    plt.subplot(1, 2, 2)
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    values = [
        results['evaluation_results']['MSE'],
        results['evaluation_results']['RMSE'],
        results['evaluation_results']['MAE'],
        results['evaluation_results']['R²']
    ]
    
    bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    plt.title('模型评估指标')
    plt.ylabel('值')
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_metrics.png'))
    plt.show()
    
    print(f"图像已保存到: {os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_metrics.png')}")
else:
    print(f"结果文件不存在: {results_path}")