"""
直接控制RL模型演示脚本
加载训练好的DQN模型，展示实际调度效果
"""

import os
import time
import torch
import numpy as np
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import *
from environment import Environment
from rl_direct_agent import DirectRLAgent
from rl_direct_controller import DirectController, MAX_ACTIONS


class DirectRLDemo:
    """直接控制RL演示器"""
    
    def __init__(self, model_path: str = "models/direct_rl_best.pt", 
                 demo_duration: float = 300.0, 
                 seed: int = 42):
        """
        初始化演示器
        
        Args:
            model_path: 模型文件路径
            demo_duration: 演示时长（秒），默认5分钟
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
            print("  将使用未训练的模型（随机策略）")
        
        print(f"演示时长: {demo_duration}秒")
        print(f"随机种子: {seed}")
        print()
    
    def run(self, verbose: bool = True, step_delay: float = 0.0):
        """
        运行演示
        
        Args:
            verbose: 是否打印详细信息
            step_delay: 每步延迟（秒），方便观察
        """
        print("=" * 70)
        print("开始演示 - 基于DQN的AGV调度系统")
        print("=" * 70)
        
        # 重置环境
        obs = self.env.reset()
        state = self.controller.get_state()
        action_list, action_mask = self.controller.build_actions()
        
        # 统计信息
        total_decisions = 0
        action_counts = {'noop': 0, 'assign_loading': 0, 'assign_unloading': 0}
        
        # 决策计时
        next_decision_time = 0.0
        next_print_time = 0.0
        print_interval = 30.0  # 每30秒打印一次状态
        
        start_time = time.time()
        
        while self.env.current_time < self.demo_duration:
            is_decision_time = (self.env.current_time >= next_decision_time)
            
            high_level_action = None
            
            if is_decision_time:
                # 选择动作（贪心，不探索）
                action_idx = self.agent.select_action(state, action_mask)
                
                # 解码动作
                high_level_action = self.controller.decode_action(action_idx)
                
                # 统计动作
                if action_idx < len(self.controller.action_list):
                    action_type = self.controller.action_list[action_idx]['type']
                    action_counts[action_type] = action_counts.get(action_type, 0) + 1
                
                total_decisions += 1
                next_decision_time = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
                
                # 打印动作（可选）
                if verbose and high_level_action is not None:
                    self._print_action(high_level_action)
            
            # 执行环境step
            next_obs, reward, done = self.env.step(high_level_action, low_level_actions=None)
            
            # 更新状态
            state = self.controller.get_state()
            action_list, action_mask = self.controller.build_actions()
            
            # 定期打印状态
            if verbose and self.env.current_time >= next_print_time:
                self._print_status()
                next_print_time = self.env.current_time + print_interval
            
            # 延迟（可选）
            if step_delay > 0:
                time.sleep(step_delay)
            
            if done:
                break
        
        real_time = time.time() - start_time
        
        # 打印最终结果
        print()
        print("=" * 70)
        print("演示结束 - 结果统计")
        print("=" * 70)
        print(f"仿真时长: {self.env.current_time:.1f}秒 ({self.env.current_time/60:.1f}分钟)")
        print(f"实际耗时: {real_time:.2f}秒")
        print(f"加速比: {self.env.current_time/real_time:.1f}x")
        print()
        print(f"【性能指标】")
        print(f"  完成货物数: {self.env.completed_cargos}")
        print(f"  超时货物数: {self.env.timed_out_cargos}")
        print(f"  吞吐率: {self.env.completed_cargos / (self.env.current_time/3600):.1f} 件/小时")
        if self.env.completed_cargos > 0:
            print(f"  平均等待时间: {self.env.total_wait_time / self.env.completed_cargos:.1f}秒")
        print()
        print(f"【决策统计】")
        print(f"  总决策次数: {total_decisions}")
        print(f"  NOOP次数: {action_counts.get('noop', 0)}")
        print(f"  上料分配: {action_counts.get('assign_loading', 0)}")
        print(f"  下料分配: {action_counts.get('assign_unloading', 0)}")
        
        return {
            'completed': self.env.completed_cargos,
            'timeout': self.env.timed_out_cargos,
            'total_wait_time': self.env.total_wait_time,
            'decisions': total_decisions			
        }
    
    def _print_action(self, action: dict):
        """打印动作信息"""
        action_type = action.get('type')
        if action_type == 'assign_loading':
            cargo_id = action.get('cargo_id')
            vehicle_id = action.get('vehicle_id')
            slot_idx = action.get('slot_idx')
            print(f"  [决策] 分配上料: 货物{cargo_id} → 车辆{vehicle_id}工位{slot_idx}")
        elif action_type == 'assign_unloading':
            cargo_id = action.get('cargo_id')
            station_id = action.get('unloading_station_id')
            print(f"  [决策] 分配下料: 货物{cargo_id} → 下料口{station_id}")
    
    def _print_status(self):
        """打印当前系统状态"""
        print()
        print(f"─── 时间: {self.env.current_time:.1f}秒 ───")
        
        # 车辆状态
        for vid, vehicle in self.env.vehicles.items():
            slots_info = []
            for i, cargo_id in enumerate(vehicle.slots):
                if cargo_id is not None:
                    slots_info.append(f"工位{i}:货物{cargo_id}")
                else:
                    slots_info.append(f"工位{i}:空")
            status = "上下料中" if vehicle.is_loading_unloading else "运行中"
            print(f"  车辆{vid}: 位置={vehicle.position:.1f}m, 速度={vehicle.velocity:.1f}m/s, {status}, [{', '.join(slots_info)}]")
        
        # 等待货物
        waiting_cargos = [c for c in self.env.cargos.values() 
                        if c.current_location.startswith("IP_") and c.completion_time is None]
        if waiting_cargos:
            print(f"  等待取货: {len(waiting_cargos)}件")
        
        print(f"  已完成: {self.env.completed_cargos}件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='直接控制RL模型演示')
    parser.add_argument('--model', type=str, default='models/direct_rl_best.pt', 
                        help='模型文件路径')
    parser.add_argument('--duration', type=float, default=300.0,
                        help='演示时长（秒）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，减少输出')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='每步延迟（秒）')
    args = parser.parse_args()
    
    demo = DirectRLDemo(
        model_path=args.model,
        demo_duration=args.duration,
        seed=args.seed
    )
    
    demo.run(verbose=not args.quiet, step_delay=args.delay)


if __name__ == "__main__":
    main()
