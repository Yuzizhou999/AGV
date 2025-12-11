"""
快速训练脚本：用于快速演示学习效果
"""

import torch
import numpy as np
from config import *
from train import TrainingManager

def quick_train():
    """运行快速训练"""
    print("\n" + "=" * 80)
    print("快速训练演示 - 10个Episodes")
    print("=" * 80)
    print("这个演示将训练模型10个episode，展示学习效果\n")
    
    # 创建训练管理器（仅10个episode）
    manager = TrainingManager(num_episodes=10, use_gpu=False)
    
    # 运行训练
    manager.train()
    
    print("\n" + "=" * 80)
    print("快速训练完成！")
    print("=" * 80)
    print("\n提示:")
    print("1. 如需完整训练，请运行: python train.py")
    print("2. 训练过程会生成模型和统计文件到 models/ 目录")
    print("3. 可以修改 config.py 中的 NUM_EPISODES 来调整训练轮数")
    print()

if __name__ == "__main__":
    quick_train()
