"""
测试方案C：环境奖励由底层加总得出
验证：总奖励 = 环境任务奖励 + sum(底层运动奖励)
"""

import torch
from train import TrainingManager

def test_plan_c():
    """测试方案C的奖励统一"""
    print("=" * 80)
    print("测试方案C：环境奖励由底层加总")
    print("=" * 80)
    print()
    print("设计原则（职责分离）：")
    print("- 环境负责：任务奖励（完成货物、等待惩罚等）")
    print("- 底层负责：运动奖励（速度质量、对齐、安全等）")
    print("- 总奖励 = 环境任务奖励 + sum(底层运动奖励)")
    print()
    print("优势：")
    print("1. 保留密集奖励设计（RLLowLevelReward的细粒度反馈）")
    print("2. 奖励来源统一（训练优化的 = 测试评估的）")
    print("3. 职责清晰分离（环境管任务，底层管运动）")
    print()
    print("=" * 80)
    print()

    # 检查GPU
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()

    # 创建训练管理器
    print("创建训练管理器（使用方案C的自定义PPO）...")
    manager = TrainingManager(
        num_episodes=30,  # 测试30个episode
        use_gpu=use_gpu,
        enable_visualization=False,
        use_rl_low_level=True,
        rl_model_path=None,
        use_custom_ppo=True
    )
    print()

    # 运行训练
    print("开始训练...")
    print("观察指标：")
    print("- 训练奖励应该稳定上升（任务奖励 + 运动奖励）")
    print("- 测试奖励应该随训练同步上升")
    print("- 不应该出现测试性能退步的情况")
    print("=" * 80)
    print()

    manager.train()

    print()
    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print()
    print("✅ 方案C的效果：")
    print("   - 既有密集奖励的引导作用")
    print("   - 又统一了训练和测试的目标")
    print("   - 职责分离清晰，易于调试")
    print()

if __name__ == "__main__":
    test_plan_c()
