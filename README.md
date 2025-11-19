# CNN医疗保险成本预测 - GPU版本

## 项目概述
本项目使用CNN模型对医疗保险数据进行回归分析，利用GPU加速训练，预测年度医疗成本。

## 系统要求
- NVIDIA GPU (支持CUDA)
- Python 3.8+
- PyTorch 2.0+ (CUDA支持)
- CUDA 11.8+

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 检查CUDA环境
```bash
python check_cuda.py
```

### 2. 运行GPU训练
```bash
# 标准训练 (20轮)
python train_cnn_gpu.py

# 快速训练 (5轮)
python train_cnn_gpu_quick.py

# 或使用批处理文件
run_cnn_gpu.bat
```

## 文件说明

### 核心模块
- `cnn_config.py`: CNN模型配置类
- `data_processor.py`: 数据预处理模块
- `cnn_trainer.py`: CNN模型训练器
- `data_loader.py`: 数据加载器

### 训练脚本
- `train_cnn_gpu.py`: 标准GPU训练脚本
- `train_cnn_gpu_quick.py`: 快速GPU训练脚本

### 结果输出
- `output/cnn_results/cnn_gpu_model.pth`: 训练好的GPU模型
- `output/cnn_results/cnn_gpu_evaluation.json`: 模型评估结果
- `output/cnn_results/cnn_gpu_training_history.png`: 训练历史图表

## 模型架构
- **输入形状**: (1, 4, 13) - 将52个特征重塑为4×13的矩阵
- **卷积层**: 2层，滤波器数量[16, 32]，卷积核大小[5, 3]
- **全连接层**: 4层，神经元数量[128, 64, 32, 1]
- **激活函数**: ReLU
- **Dropout率**: 0.2
- **优化器**: Adam (学习率=0.001)

## 性能指标
- **MSE**: 3,368.82
- **RMSE**: 58.04
- **MAE**: 43.42
- **R²**: 0.980

## 数据集
- **数据来源**: medical_insurance.csv
- **数据规模**: 100,000条记录
- **特征数量**: 52个特征 (10个分类特征，42个数值特征)
- **目标变量**: 年度医疗成本
- **数据分割**: 训练集80%，验证集10%，测试集10%