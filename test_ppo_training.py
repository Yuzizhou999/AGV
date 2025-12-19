"""
测试PPO训练功能的脚本
运行少量episode来验证训练流程是否正常工作
"""

import torch
from train import TrainingManager

def test_ppo_training():
    """测试PPO训练功能"""
    print("=" * 80)
    print("测试PPO底层控制器训练")
    print("=" * 80)
    print()
    
    # 检查GPU
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()
    
    # 创建训练管理器 - 使用RL底层控制器
    print("创建训练管理器（使用PPO底层控制器）...")
    manager = TrainingManager(
        num_episodes=20,  # 测试20个episode
        use_gpu=use_gpu,
        enable_visualization=False,
        use_rl_low_level=True,  # 启用RL控制器
        rl_model_path=None  # 从头开始训练
    )
    print()
    
    # 运行训练
    print("开始训练...")
    print("将在第10和第20个episode后进行PPO模型更新")
    print("=" * 80)
    print()
    
    manager.train()
    
    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print()
    print("✅ 如果看到 '[PPO训练]' 和 '[PPO训练完成]' 的消息，说明训练功能正常")
    print("✅ 模型应该已保存到 models/ 目录")
    print()

if __name__ == "__main__":
    test_ppo_training()
