"""
带可视化的直接控制RL模型演示脚本
实时显示环形轨道、车辆运动、货物状态
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import *
from environment import Environment
from rl_direct_agent import DirectRLAgent
from rl_direct_controller import DirectController, MAX_ACTIONS
from visualizer import AGVVisualizer


class VisualDirectRLDemo:
    """带可视化的直接控制RL演示器"""
    
    def __init__(self, model_path: str = "models/direct_rl_best.pt", 
                 demo_duration: float = 300.0, 
                 seed: int = 42):
        """
        初始化演示器
        
        Args:
            model_path: 模型文件路径
            demo_duration: 演示时长（秒）
            seed: 随机种子
        """
        self.demo_duration = demo_duration
        
        # 创建环境和控制器
        self.env = Environment(seed=seed)
        self.controller = DirectController(self.env)
        
        # 获取状态维度
        state_dim = self.controller.get_state_dim()
        
        # 创建并加载智能体
        self.agent = DirectRLAgent(
            state_dim=state_dim,
            max_actions=MAX_ACTIONS,
            device='cpu',
            epsilon_start=0.0,  # 演示时不探索
            epsilon_end=0.0
        )
        
        # 加载模型
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"✓ 已加载模型: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            print("  将使用未训练的模型")
        
        # 创建可视化器
        self.visualizer = AGVVisualizer(self.env)
        
        print(f"演示时长: {demo_duration}秒")
        print(f"随机种子: {seed}")
        print("按 Ctrl+C 可提前结束演示")
        print()
    
    def run(self, update_interval: float = 1.0, speed_multiplier: float = 1.0):
        """
        运行带可视化的演示
        
        Args:
            update_interval: 可视化更新间隔（仿真秒）
            speed_multiplier: 仿真速度倍率（1.0=实时，2.0=2倍速）
        """
        print("=" * 60)
        print("开始可视化演示 - 基于DQN的AGV调度系统")
        print("=" * 60)
        
        # 重置环境
        obs = self.env.reset()
        state = self.controller.get_state()
        action_list, action_mask = self.controller.build_actions()
        
        # 决策计时
        next_decision_time = 0.0
        next_visual_update = 0.0
        
        # 实时延迟计算
        real_step_delay = (LOW_LEVEL_CONTROL_INTERVAL / speed_multiplier) if speed_multiplier > 0 else 0
        
        try:
            while self.env.current_time < self.demo_duration:
                is_decision_time = (self.env.current_time >= next_decision_time)
                
                high_level_action = None
                
                if is_decision_time:
                    # 选择动作
                    action_idx = self.agent.select_action(state, action_mask)
                    high_level_action = self.controller.decode_action(action_idx)
                    next_decision_time = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
                
                # 执行环境step
                next_obs, reward, done = self.env.step(high_level_action, low_level_actions=None)
                
                # 更新状态
                state = self.controller.get_state()
                action_list, action_mask = self.controller.build_actions()
                
                # 更新可视化
                if self.env.current_time >= next_visual_update:
                    self.visualizer.update()
                    next_visual_update = self.env.current_time + update_interval
                
                # 实时延迟（模拟真实速度）
                if real_step_delay > 0:
                    time.sleep(real_step_delay)
                
                if done:
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断演示")
        
        # 最终更新
        self.visualizer.update()
        
        # 打印结果
        print()
        print("=" * 60)
        print("演示结束 - 结果统计")
        print("=" * 60)
        print(f"仿真时长: {self.env.current_time:.1f}秒")
        print(f"完成货物数: {self.env.completed_cargos}")
        print(f"超时货物数: {self.env.timed_out_cargos}")
        print(f"吞吐率: {self.env.completed_cargos / (self.env.current_time/3600):.1f} 件/小时")
        if self.env.completed_cargos > 0:
            print(f"平均等待时间: {self.env.total_wait_time / self.env.completed_cargos:.1f}秒")
        
        return self.visualizer
    
    def show_statistics(self, save_path: str = None):
        """显示统计图表"""
        self.visualizer.plot_statistics(save_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='带可视化的直接控制RL模型演示')
    parser.add_argument('--model', type=str, default='models/direct_rl_best.pt', 
                        help='模型文件路径')
    parser.add_argument('--duration', type=float, default=300.0,
                        help='演示时长（秒）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--speed', type=float, default=10.0,
                        help='仿真速度倍率（默认10倍速）')
    parser.add_argument('--update-interval', type=float, default=5.0,
                        help='可视化更新间隔（仿真秒）')
    parser.add_argument('--save-stats', type=str, default=None,
                        help='保存统计图表的路径')
    args = parser.parse_args()
    
    # 创建演示器
    demo = VisualDirectRLDemo(
        model_path=args.model,
        demo_duration=args.duration,
        seed=args.seed
    )
    
    # 运行演示
    visualizer = demo.run(
        update_interval=args.update_interval,
        speed_multiplier=args.speed
    )
    
    # 显示统计图表
    demo.show_statistics(args.save_stats)
    
    # 保持窗口打开
    plt.show()


if __name__ == "__main__":
    main()
