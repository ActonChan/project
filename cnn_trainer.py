# -*- coding: utf-8 -*-
"""
CNN模型训练器
负责模型训练、验证、评估等操作
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Dict

from cnn_config import CNNConfig
from cnn_model_definition import CNNModel


class CNNTrainer:
    """
    CNN模型训练器
    负责模型训练、验证、评估等操作
    """
    
    def __init__(self, config: CNNConfig, device: str = 'auto'):
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
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                try:
                    # 尝试使用CUDA
                    device_obj = torch.device('cuda')
                    # 测试是否真的可以使用CUDA
                    torch.tensor([1.0]).to(device_obj)
                    print(f"使用设备: {device_obj}")
                    return device_obj
                except Exception as e:
                    print(f"CUDA不可用，错误: {e}")
                    print("将使用CPU进行训练")
                    return torch.device('cpu')
            else:
                print("CUDA不可用，将使用CPU进行训练")
                return torch.device('cpu')
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