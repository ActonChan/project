# -*- coding: utf-8 -*-
"""
比较不同模型的性能
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取增强版CNN模型结果
enhanced_results_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_evaluation.json')

# 读取原始CNN模型结果（如果存在）
original_results_path = os.path.join(os.getcwd(), 'output', 'cnn_results', 'cnn_gpu_evaluation.json')

# 准备数据
models = []
mse_values = []
rmse_values = []
mae_values = []
r2_values = []

# 读取增强版CNN模型结果
if os.path.exists(enhanced_results_path):
    with open(enhanced_results_path, 'r', encoding='utf-8') as f:
        enhanced_results = json.load(f)
    
    models.append('增强版CNN')
    mse_values.append(enhanced_results['evaluation_results']['MSE'])
    rmse_values.append(enhanced_results['evaluation_results']['RMSE'])
    mae_values.append(enhanced_results['evaluation_results']['MAE'])
    r2_values.append(enhanced_results['evaluation_results']['R²'])
    
    print("增强版CNN模型结果:")
    print(f"  - MSE: {enhanced_results['evaluation_results']['MSE']:.2f}")
    print(f"  - RMSE: {enhanced_results['evaluation_results']['RMSE']:.2f}")
    print(f"  - MAE: {enhanced_results['evaluation_results']['MAE']:.2f}")
    print(f"  - R²: {enhanced_results['evaluation_results']['R²']:.4f}")

# 读取原始CNN模型结果
if os.path.exists(original_results_path):
    with open(original_results_path, 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    
    models.append('原始CNN')
    mse_values.append(original_results['evaluation_results']['MSE'])
    rmse_values.append(original_results['evaluation_results']['RMSE'])
    mae_values.append(original_results['evaluation_results']['MAE'])
    r2_values.append(original_results['evaluation_results']['R²'])
    
    print("\n原始CNN模型结果:")
    print(f"  - MSE: {original_results['evaluation_results']['MSE']:.2f}")
    print(f"  - RMSE: {original_results['evaluation_results']['RMSE']:.2f}")
    print(f"  - MAE: {original_results['evaluation_results']['MAE']:.2f}")
    print(f"  - R²: {original_results['evaluation_results']['R²']:.4f}")

# 如果只有一个模型，创建一个简单的比较图
if len(models) == 1:
    print("\n只有一个模型可用，无法进行比较。")
else:
    # 创建比较图表
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE比较
    axs[0, 0].bar(models, mse_values, color=['blue', 'green'])
    axs[0, 0].set_title('MSE (均方误差) 比较')
    axs[0, 0].set_ylabel('MSE')
    for i, v in enumerate(mse_values):
        axs[0, 0].text(i, v + max(mse_values)*0.01, f'{v:.2f}', ha='center')
    
    # RMSE比较
    axs[0, 1].bar(models, rmse_values, color=['blue', 'green'])
    axs[0, 1].set_title('RMSE (均方根误差) 比较')
    axs[0, 1].set_ylabel('RMSE')
    for i, v in enumerate(rmse_values):
        axs[0, 1].text(i, v + max(rmse_values)*0.01, f'{v:.2f}', ha='center')
    
    # MAE比较
    axs[1, 0].bar(models, mae_values, color=['blue', 'green'])
    axs[1, 0].set_title('MAE (平均绝对误差) 比较')
    axs[1, 0].set_ylabel('MAE')
    for i, v in enumerate(mae_values):
        axs[1, 0].text(i, v + max(mae_values)*0.01, f'{v:.2f}', ha='center')
    
    # R²比较
    axs[1, 1].bar(models, r2_values, color=['blue', 'green'])
    axs[1, 1].set_title('R² (决定系数) 比较')
    axs[1, 1].set_ylabel('R²')
    for i, v in enumerate(r2_values):
        axs[1, 1].text(i, v + max(r2_values)*0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'model_comparison.png'))
    plt.show()
    
    print(f"\n模型比较图已保存到: {os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'model_comparison.png')}")
    
    # 计算改进百分比
    if len(models) == 2:
        mse_improvement = (mse_values[1] - mse_values[0]) / mse_values[1] * 100
        rmse_improvement = (rmse_values[1] - rmse_values[0]) / rmse_values[1] * 100
        mae_improvement = (mae_values[1] - mae_values[0]) / mae_values[1] * 100
        r2_improvement = (r2_values[0] - r2_values[1]) / abs(r2_values[1]) * 100
        
        print("\n性能改进 (增强版CNN vs 原始CNN):")
        print(f"  - MSE改进: {mse_improvement:.2f}%")
        print(f"  - RMSE改进: {rmse_improvement:.2f}%")
        print(f"  - MAE改进: {mae_improvement:.2f}%")
        print(f"  - R²改进: {r2_improvement:.2f}%")