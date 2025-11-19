# -*- coding: utf-8 -*-
"""
查看训练历史图像
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 图像路径
image_path = os.path.join(os.getcwd(), 'output', 'enhanced_cnn_results', 'enhanced_cnn_training_history.png')

# 读取并显示图像
if os.path.exists(image_path):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('增强版CNN模型训练历史')
    plt.show()
else:
    print(f"图像文件不存在: {image_path}")