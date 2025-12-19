"""
测试自定义PPO训练功能
运行少量episode来验证自定义PPO是否正常工作
"""

import torch
from train import TrainingManager

def test_custom_ppo():
    """测试自定义PPO训练功能"""
    print("=" * 80)
    print("测试自定义PPO底层控制器训练")
    print("=" * 80)
    print()
    
    # 检查GPU
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()
    
    # 创建训练管理器 - 使用自定义PPO
    print("创建训练管理器（使用自定义PPO底层控制器）...")
    manager = TrainingManager(
        num_episodes=20,  # 测试20个episode
        use_gpu=use_gpu,
        enable_visualization=False,
        use_rl_low_level=True,  # 启用RL控制器
        rl_model_path=None,  # 从头开始训练
        use_custom_ppo=True  # 使用自定义PPO（不依赖SB3）
    )
    print()
    
    # 运行训练
    print("开始训练...")
    print("自定义PPO会在每个episode结束后自动更新策略")
    print("不需要额外的环境交互，训练速度更快！")
    print("=" * 80)
    print()
    
    manager.train()
    
    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print()
    print("✅ 自定义PPO的优势：")
    print("   1. 不依赖Stable-Baselines3和Gym环境")
    print("   2. 直接从环境交互中学习，无需额外步数")
    print("   3. 每个episode结束后自动更新策略")
    print("   4. 完全适配多智能体环境")
    print("   5. 训练速度更快，代码更简洁")
    print()
    print(f"✅ 模型已保存到 models/ 目录（.pth格式）")
    print()

if __name__ == "__main__":
    test_custom_ppo()
