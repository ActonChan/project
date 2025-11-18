# -*- coding: utf-8 -*-
"""
CNN回归预测模型
基于1D-CNN的医疗保险费用回归预测模型

Features:
- 1D-CNN架构用于表格数据特征提取
- 多层卷积和池化层
- Dropout正则化防止过拟合
- 完整的训练、验证、评估流程
- 可视化训练过程和结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入TensorFlow，如果失败则使用备用的sklearn MLP
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow可用，将使用CNN模型")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow不可用，将使用MLP作为备用方案")
    tf = None
    keras = None


class CNNRegressionModel:
    """
    基于1D-CNN的回归预测模型类
    专门用于处理表格数据的回归任务
    """
    
    def __init__(self, config=None):
        """
        初始化CNN回归模型
        
        Parameters:
            config: 配置管理器对象，包含模型参数和输出路径
        """
        self.config = config
        self.model = None
        self.history = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.input_shape = None
        self.feature_names = None
        self.use_tensorflow = TENSORFLOW_AVAILABLE
        
        if self.use_tensorflow:
            # CNN模型架构参数
            self.model_params = {
                'filters_1': 64,      # 第一层卷积核数量
                'filters_2': 128,     # 第二层卷积核数量
                'filters_3': 256,     # 第三层卷积核数量
                'kernel_size_1': 3,   # 第一层卷积核大小
                'kernel_size_2': 5,   # 第二层卷积核大小
                'kernel_size_3': 7,   # 第三层卷积核大小
                'pool_size': 2,       # 池化大小
                'dense_1': 128,       # 第一层全连接神经元数量
                'dense_2': 64,        # 第二层全连接神经元数量
                'dropout_rate': 0.3,  # Dropout比例
                'activation': 'relu', # 激活函数
                'output_activation': 'linear' # 输出层激活函数
            }
            
            # 训练参数
            self.training_params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 15,
                'reduce_lr_patience': 8,
                'reduce_lr_factor': 0.5
            }
        else:
            # MLP模型参数（备用方案）
            self.model_params = {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive'
            }
            
            self.training_params = {
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'n_iter_no_change': 20
            }
        
        # 获取输出路径
        if config and hasattr(config, 'output_config'):
            self.output_dir = getattr(config.output_config, 'cnn_result_dir', './output/cnn_results/')
        else:
            self.output_dir = './output/cnn_results/'
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"{model_type}回归模型初始化完成")
    
    def create_model(self, input_dim: int):
        """
        创建CNN回归模型架构
        
        Parameters:
            input_dim (int): 输入特征维度
            
        Returns:
            编译后的模型 (TensorFlow/Keras模型 或 sklearn MLPRegressor)
        """
        if self.use_tensorflow:
            # TensorFlow CNN实现
            # 确保输入数据为正确的形状 (batch_size, features, 1)
            inputs = layers.Input(shape=(input_dim, 1))
            
            # 第一个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_1'],
                kernel_size=self.model_params['kernel_size_1'],
                padding='same',
                activation=self.model_params['activation']
            )(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=self.model_params['pool_size'])(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 第二个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_2'],
                kernel_size=self.model_params['kernel_size_2'],
                padding='same',
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=self.model_params['pool_size'])(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 第三个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_3'],
                kernel_size=self.model_params['kernel_size_3'],
                padding='same',
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)  # 全局平均池化
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 全连接层
            x = layers.Dense(
                self.model_params['dense_1'],
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            x = layers.Dense(
                self.model_params['dense_2'],
                activation=self.model_params['activation']
            )(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 输出层 (回归)
            outputs = layers.Dense(
                1,
                activation=self.model_params['output_activation']
            )(x)
            
            # 创建模型
            model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Regression_Model')
            
            # 编译模型
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.training_params['learning_rate']),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.input_shape = (input_dim, 1)
            logger.info(f"CNN模型创建完成，输入形状: {self.input_shape}")
            logger.info(f"模型参数数量: {model.count_params():,}")
            
        else:
            # sklearn MLP实现
            model = MLPRegressor(
                hidden_layer_sizes=self.model_params['hidden_layer_sizes'],
                activation=self.model_params['activation'],
                solver=self.model_params['solver'],
                alpha=self.model_params['alpha'],
                learning_rate=self.model_params['learning_rate'],
                max_iter=self.training_params['max_iter'],
                random_state=self.training_params['random_state'],
                early_stopping=self.training_params['early_stopping'],
                validation_fraction=self.training_params['validation_fraction'],
                n_iter_no_change=self.training_params['n_iter_no_change'],
                verbose=True
            )
            
            self.input_shape = (input_dim,)
            logger.info(f"MLP模型创建完成，输入形状: {self.input_shape}")
            
        return model
    
    def get_hyperparameters(self, config=None):
        """
        获取模型超参数配置
        
        Parameters:
            config: 配置管理器对象
            
        Returns:
            dict: 超参数字典
        """
        hyperparameters = {
            'model_params': self.model_params,
            'training_params': self.training_params,
            'model_architecture': '1D-CNN with 3 conv blocks + global pooling + dense layers',
            'input_type': 'tabular_data_reshaped',
            'output_type': 'continuous_regression'
        }
        
        # 尝试从config获取更具体的参数
        if config and hasattr(config, 'model_config'):
            try:
                cnn_config = getattr(config.model_config, 'cnn_config', {})
                if cnn_config:
                    hyperparameters.update(cnn_config)
            except:
                pass
                
        return hyperparameters
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        数据预处理：标准化、分割、reshape
        
        Parameters:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) 预处理后的数据
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # 标准化特征
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # 标准化目标变量
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        if self.use_tensorflow:
            # TensorFlow CNN: Reshape为3D格式 (samples, features, 1)
            X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        else:
            # sklearn MLP: 使用标准2D格式
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"{model_type}数据预处理完成: 训练集{X_train_processed.shape}, 测试集{X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train_scaled, y_test_scaled
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        训练CNN模型
        
        Parameters:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            dict: 训练结果摘要
        """
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"开始训练{model_type}回归模型...")
        start_time = time.time()
        
        # 预处理数据
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        
        # 创建模型
        self.model = self.create_model(X_train.shape[1])
        
        if self.use_tensorflow:
            # TensorFlow CNN训练流程
            # 设置回调函数
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_params['early_stopping_patience'],
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.training_params['reduce_lr_factor'],
                    patience=self.training_params['reduce_lr_patience'],
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.output_dir, 'best_cnn_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # 训练模型
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.training_params['batch_size'],
                epochs=self.training_params['epochs'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # 记录训练结果
            train_results = {
                'model_name': 'CNN Regression',
                'training_time': training_time,
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_train_mae': float(self.history.history['mae'][-1]),
                'final_val_mae': float(self.history.history['val_mae'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'best_epoch': int(np.argmin(self.history.history['val_loss']) + 1)
            }
            
        else:
            # sklearn MLP训练流程
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # MLP没有history对象，创建模拟的历史数据
            train_results = {
                'model_name': 'MLP Regression',
                'training_time': training_time,
                'epochs_trained': self.model.n_iter_,
                'final_train_loss': float(self.model.loss_),
                'convergence_achieved': self.model.n_iter_ < self.training_params['max_iter']
            }
            
            # 创建历史对象以保持一致性
            self.history = type('History', (), {
                'history': {
                    'loss': [self.model.loss_] * self.model.n_iter_,
                    'val_loss': [self.model.loss_] * self.model.n_iter_
                }
            })()
        
        logger.info(f"{model_type}模型训练完成，耗时: {training_time:.2f}秒")
        
        if self.use_tensorflow:
            logger.info(f"最佳验证损失: {train_results['best_val_loss']:.6f}")
        else:
            logger.info(f"最终损失: {train_results['final_train_loss']:.6f}")
        
        return train_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Parameters:
            X (pd.DataFrame): 输入特征
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 预处理输入数据
        X_scaled = self.scaler_X.transform(X)
        
        if self.use_tensorflow:
            # TensorFlow CNN: 需要reshape
            X_processed = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            # 进行预测
            predictions_scaled = self.model.predict(X_processed, verbose=0)
        else:
            # sklearn MLP: 直接使用2D数据
            predictions_scaled = self.model.predict(X_scaled)
        
        # 反标准化预测结果
        if self.use_tensorflow:
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        else:
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        评估模型性能
        
        Parameters:
            X_test (pd.DataFrame): 测试特征
            y_test (pd.Series): 测试目标
            
        Returns:
            dict: 评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 进行预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 计算解释方差
        explained_variance = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'explained_variance': float(explained_variance)
        }
        
        logger.info(f"模型评估完成 - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """
        创建模型评估可视化图表
        
        Parameters:
            X_test (pd.DataFrame): 测试特征
            y_test (pd.Series): 测试目标
            save_path (str): 保存路径
        """
        if save_path is None:
            save_path = self.output_dir
            
        # 进行预测
        y_pred = self.predict(X_test)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        model_type = "CNN" if self.use_tensorflow else "MLP"
        fig.suptitle(f'{model_type}回归模型评估结果', fontsize=16, fontweight='bold')
        
        # 1. 真实值 vs 预测值散点图
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=30)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('真实值 vs 预测值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='orange', s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练历史
        if self.use_tensorflow and self.history:
            epochs = range(1, len(self.history.history['loss']) + 1)
            axes[1, 0].plot(epochs, self.history.history['loss'], 'b-', label='训练损失', linewidth=2)
            if 'val_loss' in self.history.history:
                axes[1, 0].plot(epochs, self.history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].set_title('训练历史')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 对于MLP模型显示参数信息
            axes[1, 0].text(0.5, 0.5, f'{model_type}模型\n参数数量: {self.model.n_parameters_ if hasattr(self.model, "n_parameters_") else "N/A"}', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('模型信息')
            axes[1, 0].axis('off')
        
        # 4. 预测误差分布
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('预测误差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('预测误差分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(save_path, f'{model_type.lower()}_evaluation_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"评估图表已保存: {plot_path}")
        
        plt.show()
    
    def save_model_results(self, train_results: Dict, eval_metrics: Dict, save_path: str = None):
        """
        保存模型结果到文件
        
        Parameters:
            train_results (Dict): 训练结果
            eval_metrics (Dict): 评估指标
            save_path (str): 保存路径
        """
        if save_path is None:
            save_path = self.output_dir
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
            
        # 合并所有结果
        all_results = {
            'model_info': {
                'model_name': f'{model_type} Regression Model',
                'model_type': '1D-Convolutional Neural Network' if self.use_tensorflow else 'Multi-layer Perceptron',
                'input_type': 'tabular_data',
                'output_type': 'continuous_regression',
                'architecture': '3-conv-blocks + global-pooling + dense-layers' if self.use_tensorflow else '3-hidden-layers MLP',
                'framework': 'TensorFlow/Keras' if self.use_tensorflow else 'scikit-learn'
            },
            'hyperparameters': self.get_hyperparameters(),
            'training_results': train_results,
            'evaluation_metrics': eval_metrics,
            'feature_info': {
                'num_features': len(self.feature_names) if self.feature_names else None,
                'feature_names': self.feature_names
            }
        }
        
        # 保存JSON文件
        json_path = os.path.join(save_path, f'{model_type.lower()}_model_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存评估指标为CSV
        metrics_df = pd.DataFrame([eval_metrics])
        csv_path = os.path.join(save_path, f'{model_type.lower()}_evaluation_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        # 保存特征重要性（如果可用且使用TensorFlow）
        if self.use_tensorflow and hasattr(self.model, 'get_weights'):
            try:
                feature_importance = np.abs(self.model.layers[1].get_weights()[0]).mean(axis=(0, 2))
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = os.path.join(save_path, f'{model_type.lower()}_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
            except:
                pass  # 如果无法计算特征重要性，忽略
        
        logger.info(f"{model_type}模型结果已保存到: {save_path}")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        if self.model is None:
            return {'error': '模型尚未创建或训练'}
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        
        if self.use_tensorflow:
            # TensorFlow模型信息
            info = {
                'model_name': self.model.name,
                'input_shape': self.input_shape,
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'model_summary': str(self.model.summary())
            }
        else:
            # MLP模型信息
            info = {
                'model_name': f'MLPRegressor (sklearn)',
                'input_shape': (self.input_shape,) if isinstance(self.input_shape, int) else self.input_shape,
                'n_features_in_': getattr(self.model, 'n_features_in_', None),
                'n_layers_': getattr(self.model, 'n_layers_', None),
                'hidden_layer_sizes': getattr(self.model, 'hidden_layer_sizes', None),
                'activation': getattr(self.model, 'activation', None),
                'solver': getattr(self.model, 'solver', None),
                'alpha': getattr(self.model, 'alpha', None),
                'learning_rate_init': getattr(self.model, 'learning_rate_init', None),
                'max_iter': getattr(self.model, 'max_iter', None),
                'model_summary': str(self.model)
            }
        
        return info


def main():
    """主函数示例"""
    # 示例用法
    logger.info("CNN回归模型模块加载完成")
    
    # 这里可以添加实际的使用示例
    # 例如：加载数据、训练模型、评估结果等


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
CNN回归预测模型
基于1D-CNN的医疗保险费用回归预测模型

Features:
- 1D-CNN架构用于表格数据特征提取
- 多层卷积和池化层
- Dropout正则化防止过拟合
- 完整的训练、验证、评估流程
- 可视化训练过程和结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入TensorFlow，如果失败则使用备用的sklearn MLP
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow可用，将使用CNN模型")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow不可用，将使用MLP作为备用方案")
    tf = None
    keras = None


class CNNRegressionModel:
    """
    基于1D-CNN的回归预测模型类
    专门用于处理表格数据的回归任务
    """
    
    def __init__(self, config=None):
        """
        初始化CNN回归模型
        
        Parameters:
            config: 配置管理器对象，包含模型参数和输出路径
        """
        self.config = config
        self.model = None
        self.history = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.input_shape = None
        self.feature_names = None
        self.use_tensorflow = TENSORFLOW_AVAILABLE
        
        if self.use_tensorflow:
            # CNN模型架构参数
            self.model_params = {
                'filters_1': 64,      # 第一层卷积核数量
                'filters_2': 128,     # 第二层卷积核数量
                'filters_3': 256,     # 第三层卷积核数量
                'kernel_size_1': 3,   # 第一层卷积核大小
                'kernel_size_2': 5,   # 第二层卷积核大小
                'kernel_size_3': 7,   # 第三层卷积核大小
                'pool_size': 2,       # 池化大小
                'dense_1': 128,       # 第一层全连接神经元数量
                'dense_2': 64,        # 第二层全连接神经元数量
                'dropout_rate': 0.3,  # Dropout比例
                'activation': 'relu', # 激活函数
                'output_activation': 'linear' # 输出层激活函数
            }
            
            # 训练参数
            self.training_params = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 15,
                'reduce_lr_patience': 8,
                'reduce_lr_factor': 0.5
            }
        else:
            # MLP模型参数（备用方案）
            self.model_params = {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive'
            }
            
            self.training_params = {
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'n_iter_no_change': 20
            }
        
        # 获取输出路径
        if config and hasattr(config, 'output_config'):
            self.output_dir = getattr(config.output_config, 'cnn_result_dir', './output/cnn_results/')
        else:
            self.output_dir = './output/cnn_results/'
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"{model_type}回归模型初始化完成")
    
    def create_model(self, input_dim: int):
        """
        创建CNN回归模型架构
        
        Parameters:
            input_dim (int): 输入特征维度
            
        Returns:
            编译后的模型 (TensorFlow/Keras模型 或 sklearn MLPRegressor)
        """
        if self.use_tensorflow:
            # TensorFlow CNN实现
            # 确保输入数据为正确的形状 (batch_size, features, 1)
            inputs = layers.Input(shape=(input_dim, 1))
            
            # 第一个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_1'],
                kernel_size=self.model_params['kernel_size_1'],
                padding='same',
                activation=self.model_params['activation']
            )(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=self.model_params['pool_size'])(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 第二个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_2'],
                kernel_size=self.model_params['kernel_size_2'],
                padding='same',
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=self.model_params['pool_size'])(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 第三个卷积块
            x = layers.Conv1D(
                filters=self.model_params['filters_3'],
                kernel_size=self.model_params['kernel_size_3'],
                padding='same',
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)  # 全局平均池化
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 全连接层
            x = layers.Dense(
                self.model_params['dense_1'],
                activation=self.model_params['activation']
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            x = layers.Dense(
                self.model_params['dense_2'],
                activation=self.model_params['activation']
            )(x)
            x = layers.Dropout(self.model_params['dropout_rate'])(x)
            
            # 输出层 (回归)
            outputs = layers.Dense(
                1,
                activation=self.model_params['output_activation']
            )(x)
            
            # 创建模型
            model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Regression_Model')
            
            # 编译模型
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.training_params['learning_rate']),
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.input_shape = (input_dim, 1)
            logger.info(f"CNN模型创建完成，输入形状: {self.input_shape}")
            logger.info(f"模型参数数量: {model.count_params():,}")
            
        else:
            # sklearn MLP实现
            model = MLPRegressor(
                hidden_layer_sizes=self.model_params['hidden_layer_sizes'],
                activation=self.model_params['activation'],
                solver=self.model_params['solver'],
                alpha=self.model_params['alpha'],
                learning_rate=self.model_params['learning_rate'],
                max_iter=self.training_params['max_iter'],
                random_state=self.training_params['random_state'],
                early_stopping=self.training_params['early_stopping'],
                validation_fraction=self.training_params['validation_fraction'],
                n_iter_no_change=self.training_params['n_iter_no_change'],
                verbose=True
            )
            
            self.input_shape = (input_dim,)
            logger.info(f"MLP模型创建完成，输入形状: {self.input_shape}")
            
        return model
    
    def get_hyperparameters(self, config=None):
        """
        获取模型超参数配置
        
        Parameters:
            config: 配置管理器对象
            
        Returns:
            dict: 超参数字典
        """
        hyperparameters = {
            'model_params': self.model_params,
            'training_params': self.training_params,
            'model_architecture': '1D-CNN with 3 conv blocks + global pooling + dense layers',
            'input_type': 'tabular_data_reshaped',
            'output_type': 'continuous_regression'
        }
        
        # 尝试从config获取更具体的参数
        if config and hasattr(config, 'model_config'):
            try:
                cnn_config = getattr(config.model_config, 'cnn_config', {})
                if cnn_config:
                    hyperparameters.update(cnn_config)
            except:
                pass
                
        return hyperparameters
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        数据预处理：标准化、分割、reshape
        
        Parameters:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) 预处理后的数据
        """
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # 标准化特征
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # 标准化目标变量
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        if self.use_tensorflow:
            # TensorFlow CNN: Reshape为3D格式 (samples, features, 1)
            X_train_processed = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_processed = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        else:
            # sklearn MLP: 使用标准2D格式
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"{model_type}数据预处理完成: 训练集{X_train_processed.shape}, 测试集{X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train_scaled, y_test_scaled
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        训练CNN模型
        
        Parameters:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 目标变量
            
        Returns:
            dict: 训练结果摘要
        """
        model_type = "CNN" if self.use_tensorflow else "MLP"
        logger.info(f"开始训练{model_type}回归模型...")
        start_time = time.time()
        
        # 预处理数据
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        
        # 创建模型
        self.model = self.create_model(X_train.shape[1])
        
        if self.use_tensorflow:
            # TensorFlow CNN训练流程
            # 设置回调函数
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_params['early_stopping_patience'],
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.training_params['reduce_lr_factor'],
                    patience=self.training_params['reduce_lr_patience'],
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.output_dir, 'best_cnn_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # 训练模型
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.training_params['batch_size'],
                epochs=self.training_params['epochs'],
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # 记录训练结果
            train_results = {
                'model_name': 'CNN Regression',
                'training_time': training_time,
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_train_mae': float(self.history.history['mae'][-1]),
                'final_val_mae': float(self.history.history['val_mae'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'best_epoch': int(np.argmin(self.history.history['val_loss']) + 1)
            }
            
        else:
            # sklearn MLP训练流程
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # MLP没有history对象，创建模拟的历史数据
            train_results = {
                'model_name': 'MLP Regression',
                'training_time': training_time,
                'epochs_trained': self.model.n_iter_,
                'final_train_loss': float(self.model.loss_),
                'convergence_achieved': self.model.n_iter_ < self.training_params['max_iter']
            }
            
            # 创建历史对象以保持一致性
            self.history = type('History', (), {
                'history': {
                    'loss': [self.model.loss_] * self.model.n_iter_,
                    'val_loss': [self.model.loss_] * self.model.n_iter_
                }
            })()
        
        logger.info(f"{model_type}模型训练完成，耗时: {training_time:.2f}秒")
        
        if self.use_tensorflow:
            logger.info(f"最佳验证损失: {train_results['best_val_loss']:.6f}")
        else:
            logger.info(f"最终损失: {train_results['final_train_loss']:.6f}")
        
        return train_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Parameters:
            X (pd.DataFrame): 输入特征
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 预处理输入数据
        X_scaled = self.scaler_X.transform(X)
        
        if self.use_tensorflow:
            # TensorFlow CNN: 需要reshape
            X_processed = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            # 进行预测
            predictions_scaled = self.model.predict(X_processed, verbose=0)
        else:
            # sklearn MLP: 直接使用2D数据
            predictions_scaled = self.model.predict(X_scaled)
        
        # 反标准化预测结果
        if self.use_tensorflow:
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        else:
            predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        评估模型性能
        
        Parameters:
            X_test (pd.DataFrame): 测试特征
            y_test (pd.Series): 测试目标
            
        Returns:
            dict: 评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 进行预测
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 计算解释方差
        explained_variance = 1 - (np.var(y_test - y_pred) / np.var(y_test))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'explained_variance': float(explained_variance)
        }
        
        logger.info(f"模型评估完成 - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """
        创建模型评估可视化图表
        
        Parameters:
            X_test (pd.DataFrame): 测试特征
            y_test (pd.Series): 测试目标
            save_path (str): 保存路径
        """
        if save_path is None:
            save_path = self.output_dir
            
        # 进行预测
        y_pred = self.predict(X_test)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        model_type = "CNN" if self.use_tensorflow else "MLP"
        fig.suptitle(f'{model_type}回归模型评估结果', fontsize=16, fontweight='bold')
        
        # 1. 真实值 vs 预测值散点图
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=30)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('真实值 vs 预测值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差图
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='orange', s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练历史
        if self.use_tensorflow and self.history:
            epochs = range(1, len(self.history.history['loss']) + 1)
            axes[1, 0].plot(epochs, self.history.history['loss'], 'b-', label='训练损失', linewidth=2)
            if 'val_loss' in self.history.history:
                axes[1, 0].plot(epochs, self.history.history['val_loss'], 'r-', label='验证损失', linewidth=2)
            axes[1, 0].set_xlabel('训练轮次')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].set_title('训练历史')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # 对于MLP模型显示参数信息
            axes[1, 0].text(0.5, 0.5, f'{model_type}模型\n参数数量: {self.model.n_parameters_ if hasattr(self.model, "n_parameters_") else "N/A"}', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('模型信息')
            axes[1, 0].axis('off')
        
        # 4. 预测误差分布
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('预测误差')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('预测误差分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(save_path, f'{model_type.lower()}_evaluation_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"评估图表已保存: {plot_path}")
        
        plt.show()
    
    def save_model_results(self, train_results: Dict, eval_metrics: Dict, save_path: str = None):
        """
        保存模型结果到文件
        
        Parameters:
            train_results (Dict): 训练结果
            eval_metrics (Dict): 评估指标
            save_path (str): 保存路径
        """
        if save_path is None:
            save_path = self.output_dir
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
            
        # 合并所有结果
        all_results = {
            'model_info': {
                'model_name': f'{model_type} Regression Model',
                'model_type': '1D-Convolutional Neural Network' if self.use_tensorflow else 'Multi-layer Perceptron',
                'input_type': 'tabular_data',
                'output_type': 'continuous_regression',
                'architecture': '3-conv-blocks + global-pooling + dense-layers' if self.use_tensorflow else '3-hidden-layers MLP',
                'framework': 'TensorFlow/Keras' if self.use_tensorflow else 'scikit-learn'
            },
            'hyperparameters': self.get_hyperparameters(),
            'training_results': train_results,
            'evaluation_metrics': eval_metrics,
            'feature_info': {
                'num_features': len(self.feature_names) if self.feature_names else None,
                'feature_names': self.feature_names
            }
        }
        
        # 保存JSON文件
        json_path = os.path.join(save_path, f'{model_type.lower()}_model_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存评估指标为CSV
        metrics_df = pd.DataFrame([eval_metrics])
        csv_path = os.path.join(save_path, f'{model_type.lower()}_evaluation_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        # 保存特征重要性（如果可用且使用TensorFlow）
        if self.use_tensorflow and hasattr(self.model, 'get_weights'):
            try:
                feature_importance = np.abs(self.model.layers[1].get_weights()[0]).mean(axis=(0, 2))
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = os.path.join(save_path, f'{model_type.lower()}_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
            except:
                pass  # 如果无法计算特征重要性，忽略
        
        logger.info(f"{model_type}模型结果已保存到: {save_path}")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        if self.model is None:
            return {'error': '模型尚未创建或训练'}
        
        model_type = "CNN" if self.use_tensorflow else "MLP"
        
        if self.use_tensorflow:
            # TensorFlow模型信息
            info = {
                'model_name': self.model.name,
                'input_shape': self.input_shape,
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'model_summary': str(self.model.summary())
            }
        else:
            # MLP模型信息
            info = {
                'model_name': f'MLPRegressor (sklearn)',
                'input_shape': (self.input_shape,) if isinstance(self.input_shape, int) else self.input_shape,
                'n_features_in_': getattr(self.model, 'n_features_in_', None),
                'n_layers_': getattr(self.model, 'n_layers_', None),
                'hidden_layer_sizes': getattr(self.model, 'hidden_layer_sizes', None),
                'activation': getattr(self.model, 'activation', None),
                'solver': getattr(self.model, 'solver', None),
                'alpha': getattr(self.model, 'alpha', None),
                'learning_rate_init': getattr(self.model, 'learning_rate_init', None),
                'max_iter': getattr(self.model, 'max_iter', None),
                'model_summary': str(self.model)
            }
        
        return info


def main():
    """主函数示例"""
    # 示例用法
    logger.info("CNN回归模型模块加载完成")
    
    # 这里可以添加实际的使用示例
    # 例如：加载数据、训练模型、评估结果等


if __name__ == "__main__":
    main()