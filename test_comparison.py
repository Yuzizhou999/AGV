"""
对比测试：自定义PPO vs 启发式控制器
运行相同episode数，对比性能和训练速度
"""

import torch
import time
from train import TrainingManager

def test_comparison():
    """对比测试"""
    print("=" * 80)
    print("对比测试：自定义PPO vs 启发式控制器")
    print("=" * 80)
    print()
    
    use_gpu = torch.cuda.is_available()
    test_episodes = 50  # 测试50个episode
    
    # ========== 测试1：启发式控制器（基线） ==========
    print("【测试1】启发式底层控制器（基线）")
    print("-" * 80)
    
    start_time = time.time()
    manager_heuristic = TrainingManager(
        num_episodes=test_episodes,
        use_gpu=use_gpu,
        use_rl_low_level=False,
        use_heuristic_low_level=True
    )
    manager_heuristic.train()
    heuristic_time = time.time() - start_time
    
    # 统计启发式性能
    heuristic_avg_reward = sum(manager_heuristic.episode_rewards) / len(manager_heuristic.episode_rewards)
    heuristic_avg_completion = sum(manager_heuristic.episode_completions) / len(manager_heuristic.episode_completions)
    
    print()
    print("=" * 80)
    
    # ========== 测试2：自定义PPO ==========
    print("【测试2】自定义PPO底层控制器")
    print("-" * 80)
    
    start_time = time.time()
    manager_ppo = TrainingManager(
        num_episodes=test_episodes,
        use_gpu=use_gpu,
        use_rl_low_level=True,
        use_custom_ppo=True
    )
    manager_ppo.train()
    ppo_time = time.time() - start_time
    
    # 统计PPO性能
    ppo_avg_reward = sum(manager_ppo.episode_rewards) / len(manager_ppo.episode_rewards)
    ppo_avg_completion = sum(manager_ppo.episode_completions) / len(manager_ppo.episode_completions)
    
    # 后期性能（最后20个episode）
    ppo_late_reward = sum(manager_ppo.episode_rewards[-20:]) / 20
    ppo_late_completion = sum(manager_ppo.episode_completions[-20:]) / 20
    
    print()
    print("=" * 80)
    print("对比结果")
    print("=" * 80)
    print()
    
    print(f"{'指标':<25} | {'启发式':<15} | {'自定义PPO':<15} | {'后期PPO':<15}")
    print("-" * 80)
    print(f"{'平均奖励':<25} | {heuristic_avg_reward:>13.2f} | {ppo_avg_reward:>13.2f} | {ppo_late_reward:>13.2f}")
    print(f"{'平均完成件数':<25} | {heuristic_avg_completion:>13.1f} | {ppo_avg_completion:>13.1f} | {ppo_late_completion:>13.1f}")
    print(f"{'训练时间(秒)':<25} | {heuristic_time:>13.1f} | {ppo_time:>13.1f} | {'-':>13}")
    print(f"{'平均每episode时间(秒)':<25} | {heuristic_time/test_episodes:>13.2f} | {ppo_time/test_episodes:>13.2f} | {'-':>13}")
    print()
    
    # 分析
    print("分析：")
    if ppo_late_reward > heuristic_avg_reward:
        improvement = (ppo_late_reward - heuristic_avg_reward) / abs(heuristic_avg_reward) * 100
        print(f"✅ PPO在后期表现优于启发式，奖励提升 {improvement:.1f}%")
    else:
        gap = (heuristic_avg_reward - ppo_late_reward) / abs(heuristic_avg_reward) * 100
        print(f"⚠️  PPO在后期仍低于启发式 {gap:.1f}%，需要更多训练")
    
    time_ratio = ppo_time / heuristic_time
    print(f"⏱️  PPO训练时间是启发式的 {time_ratio:.2f} 倍")
    
    if time_ratio < 1.5:
        print(f"✅ 时间开销可接受（不到1.5倍）")
    elif time_ratio < 2.0:
        print(f"⚠️  时间开销较大（1.5-2倍）")
    else:
        print(f"❌ 时间开销过大（超过2倍）")
    
    print()
    print("建议：")
    if ppo_late_reward < heuristic_avg_reward:
        print("1. 增加训练episode数（建议200+）")
        print("2. 调整奖励函数权重")
        print("3. 尝试不同的学习率和网络结构")
    else:
        print("1. PPO已显示出学习能力，继续训练可能有更好表现")
        print("2. 可以保存当前模型并继续训练")
        print("3. 考虑调整超参数进一步优化")
    
    print()

if __name__ == "__main__":
    test_comparison()
