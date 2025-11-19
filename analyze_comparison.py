# -*- coding: utf-8 -*-
"""
分析模型比较结果
"""

import os
import json

# 读取增强版CNN模型结果
enhanced_results_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_evaluation.json')

# 读取原始CNN模型结果
original_results_path = os.path.join(os.getcwd(), 'output', 'cnn_results', 'cnn_gpu_evaluation.json')

# 读取增强版CNN模型结果
with open(enhanced_results_path, 'r', encoding='utf-8') as f:
    enhanced_results = json.load(f)

# 读取原始CNN模型结果
with open(original_results_path, 'r', encoding='utf-8') as f:
    original_results = json.load(f)

# 提取评估指标
enhanced_mse = enhanced_results['evaluation_results']['MSE']
enhanced_rmse = enhanced_results['evaluation_results']['RMSE']
enhanced_mae = enhanced_results['evaluation_results']['MAE']
enhanced_r2 = enhanced_results['evaluation_results']['R²']

original_mse = original_results['evaluation_results']['MSE']
original_rmse = original_results['evaluation_results']['RMSE']
original_mae = original_results['evaluation_results']['MAE']
original_r2 = original_results['evaluation_results']['R²']

# 计算改进百分比
mse_improvement = (original_mse - enhanced_mse) / original_mse * 100
rmse_improvement = (original_rmse - enhanced_rmse) / original_rmse * 100
mae_improvement = (original_mae - enhanced_mae) / original_mae * 100
r2_improvement = (enhanced_r2 - original_r2) / abs(original_r2) * 100

print("=" * 60)
print("模型性能比较分析")
print("=" * 60)

print("\n1. 评估指标对比:")
print(f"{'指标':<10} {'原始CNN':<15} {'增强版CNN':<15} {'改进百分比':<10}")
print("-" * 60)
print(f"{'MSE':<10} {original_mse:<15.2f} {enhanced_mse:<15.2f} {mse_improvement:<10.2f}%")
print(f"{'RMSE':<10} {original_rmse:<15.2f} {enhanced_rmse:<15.2f} {rmse_improvement:<10.2f}%")
print(f"{'MAE':<10} {original_mae:<15.2f} {enhanced_mae:<15.2f} {mae_improvement:<10.2f}%")
print(f"{'R²':<10} {original_r2:<15.4f} {enhanced_r2:<15.4f} {r2_improvement:<10.2f}%")

print("\n2. 模型配置对比:")
print("\n原始CNN模型:")
print(f"  - 卷积层数: {original_results['model_config']['conv_layers']}")
print(f"  - 卷积核数量: {original_results['model_config']['filters']}")
print(f"  - 卷积核大小: {original_results['model_config']['kernel_sizes']}")
print(f"  - 全连接层: {original_results['model_config']['fc_layers']}")
print(f"  - 总参数数量: {original_results['evaluation_results']['model_info']['total_parameters']}")

print("\n增强版CNN模型:")
print(f"  - 卷积层数: {enhanced_results['model_config']['conv_layers']}")
print(f"  - 卷积核数量: {enhanced_results['model_config']['filters']}")
print(f"  - 卷积核大小: {enhanced_results['model_config']['kernel_sizes']}")
print(f"  - 全连接层: {enhanced_results['evaluation_results']['model_info']['fc_layers']}")
print(f"  - 总参数数量: {enhanced_results['evaluation_results']['model_info']['total_parameters']}")

print("\n3. 训练结果对比:")
print("\n原始CNN模型:")
print(f"  - 最终训练损失: {original_results['training_results']['final_train_loss']:.2f}")
print(f"  - 最终验证损失: {original_results['training_results']['final_val_loss']:.2f}")
print(f"  - 最佳验证损失: {original_results['training_results']['best_val_loss']:.2f}")

print("\n增强版CNN模型:")
print(f"  - 最终训练损失: {enhanced_results['training_results']['final_train_loss']:.2f}")
print(f"  - 最终验证损失: {enhanced_results['training_results']['final_val_loss']:.2f}")
print(f"  - 最佳验证损失: {enhanced_results['training_results']['best_val_loss']:.2f}")

print("\n4. 分析结论:")
if mse_improvement > 0:
    print(f"  - MSE改进了{mse_improvement:.2f}%，表明增强版CNN的预测误差更小")
else:
    print(f"  - MSE增加了{abs(mse_improvement):.2f}%，表明增强版CNN的预测误差更大")

if rmse_improvement > 0:
    print(f"  - RMSE改进了{rmse_improvement:.2f}%，表明增强版CNN的预测误差更小")
else:
    print(f"  - RMSE增加了{abs(rmse_improvement):.2f}%，表明增强版CNN的预测误差更大")

if mae_improvement > 0:
    print(f"  - MAE改进了{mae_improvement:.2f}%，表明增强版CNN的预测误差更小")
else:
    print(f"  - MAE增加了{abs(mae_improvement):.2f}%，表明增强版CNN的预测误差更大")

if r2_improvement > 0:
    print(f"  - R²改进了{r2_improvement:.2f}%，表明增强版CNN对数据的解释能力更强")
else:
    print(f"  - R²降低了{abs(r2_improvement):.2f}%，表明增强版CNN对数据的解释能力更弱")

# 综合评价
if mse_improvement > 0 and rmse_improvement > 0 and mae_improvement > 0 and r2_improvement > 0:
    print("\n综合评价: 增强版CNN在所有指标上都优于原始CNN模型")
elif mse_improvement > 0 and rmse_improvement > 0 and mae_improvement > 0:
    print("\n综合评价: 增强版CNN在误差指标上优于原始CNN模型")
elif r2_improvement > 0:
    print("\n综合评价: 增强版CNN在数据解释能力上优于原始CNN模型")
else:
    print("\n综合评价: 原始CNN模型在大多数指标上优于增强版CNN模型")

print("\n5. 可能的原因分析:")
print("  - 增强版CNN增加了额外的全连接层，可能导致模型过于复杂")
print("  - 增加的参数可能导致过拟合，特别是在训练数据有限的情况下")
print("  - 原始CNN模型的结构可能更适合当前的数据集")
print("  - 增强版CNN可能需要更多的训练轮数或不同的学习率来达到最佳性能")

print("\n6. 改进建议:")
print("  - 尝试减少增强版CNN的全连接层大小")
print("  - 增加正则化技术，如dropout或L2正则化")
print("  - 调整学习率和训练轮数")
print("  - 尝试不同的优化器")
print("  - 增加训练数据或使用数据增强技术")