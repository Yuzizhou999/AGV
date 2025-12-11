"""
评估脚本：8小时完整仿真评估
"""

import numpy as np
import torch
from typing import Dict, List
import time
from datetime import datetime

from config import *
from environment import Environment
from agent_high_level import HighLevelAgent, HighLevelController
from agent_low_level import LowLevelAgent, LowLevelController


class EvaluationManager:
    """评估管理器 - 8小时完整仿真"""
    
    def __init__(self, model_path: str = None, use_gpu: bool = False):
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        
        # 初始化环境（使用评估时长）
        self.env = Environment(seed=42)
        
        # 计算观测维度
        high_level_obs_dim = 100
        max_decisions = NUM_LOADING_STATIONS * 2 + MAX_VEHICLES * 2
        low_level_obs_dim = MAX_VEHICLES + 2
        low_level_action_dim = 3
        
        # 初始化智能体
        self.high_level_agent = HighLevelAgent(high_level_obs_dim, max_decisions, self.device)
        self.low_level_agent = LowLevelAgent(low_level_obs_dim, low_level_action_dim, self.device)
        
        # 加载模型（如果提供）
        if model_path:
            self.load_models(model_path)
        
        # 初始化控制器
        self.high_level_controller = HighLevelController(self.high_level_agent, self.env)
        self.low_level_controller = LowLevelController(self.low_level_agent, self.env)
        
        # 设置为评估模式（不探索）
        self.high_level_agent.epsilon = 0.0
        self.low_level_agent.epsilon = 0.0
    
    def load_models(self, path_prefix: str):
        """加载训练好的模型"""
        try:
            high_level_path = f"{path_prefix}_high_level.pt"
            low_level_path = f"{path_prefix}_low_level.pt"
            
            self.high_level_agent.q_network.load_state_dict(
                torch.load(high_level_path, map_location=self.device)
            )
            self.low_level_agent.q_network.load_state_dict(
                torch.load(low_level_path, map_location=self.device)
            )
            print(f"✓ 成功加载模型: {path_prefix}")
        except Exception as e:
            print(f"⚠ 未能加载模型: {e}")
            print("  将使用未训练的模型进行评估")
    
    def evaluate_8_hours(self, duration: float = EPISODE_DURATION_EVAL) -> Dict:
        """
        执行8小时完整仿真评估
        
        Args:
            duration: 仿真时长（秒），默认8小时
        
        Returns:
            评估结果字典
        """
        print("=" * 80)
        print("开始8小时完整仿真评估")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"仿真时长: {duration}秒 ({duration/3600:.1f}小时)")
        print(f"车辆数: {MAX_VEHICLES}")
        print(f"上料口数: {NUM_LOADING_STATIONS}")
        print(f"下料口数: {NUM_UNLOADING_STATIONS}")
        print("=" * 80)
        print()
        
        # 重置环境
        obs = self.env.reset()
        
        # 临时修改仿真时长
        original_duration = EPISODE_DURATION
        import config
        config.EPISODE_DURATION = duration
        self.env.current_time = 0.0
        
        episode_reward = 0.0
        step_count = 0
        start_time = time.time()
        
        # 统计信息
        hourly_stats = []
        last_hour_check = 0.0
        
        print("开始仿真...")
        print("-" * 80)
        
        try:
            while self.env.current_time < duration:
                # 高层决策
                task_list = self.high_level_controller.compute_action(obs)
                
                # 低层控制
                low_actions = self.low_level_controller.compute_actions()
                
                # 执行一步
                obs, reward, done = self.env.step(task_list, low_actions)
                episode_reward += reward
                step_count += 1
                
                # 每小时统计一次
                if self.env.current_time - last_hour_check >= 3600:
                    hour_num = int(self.env.current_time / 3600)
                    hourly_stats.append({
                        'hour': hour_num,
                        'completed': self.env.completed_cargos,
                        'timeout': self.env.timed_out_cargos,
                        'total_cargos': len(self.env.cargos),
                        'avg_wait': self.env.total_wait_time / max(1, self.env.cargo_counter)
                    })
                    print(f"第{hour_num}小时: 完成{self.env.completed_cargos}件, "
                          f"超时{self.env.timed_out_cargos}件, "
                          f"总货物{len(self.env.cargos)}件")
                    last_hour_check = self.env.current_time
                
                # 定期打印进度
                if step_count % 1000 == 0:
                    progress = self.env.current_time / duration * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / progress * 100 - elapsed if progress > 0 else 0
                    print(f"  进度: {progress:.1f}% | "
                          f"步数: {step_count} | "
                          f"已耗时: {elapsed:.1f}s | "
                          f"预计剩余: {eta:.1f}s", flush=True)
                
                if done:
                    break
        
        except KeyboardInterrupt:
            print("\n⚠ 用户中断评估")
        
        # 恢复原始配置
        config.EPISODE_DURATION = original_duration
        
        total_time = time.time() - start_time
        
        # 计算最终统计
        results = {
            'duration': self.env.current_time,
            'total_reward': episode_reward,
            'completed_cargos': self.env.completed_cargos,
            'timed_out_cargos': self.env.timed_out_cargos,
            'total_cargos': len(self.env.cargos),
            'completion_rate': self.env.completed_cargos / max(1, len(self.env.cargos)),
            'timeout_rate': self.env.timed_out_cargos / max(1, len(self.env.cargos)),
            'avg_wait_time': self.env.total_wait_time / max(1, self.env.cargo_counter),
            'throughput_per_hour': self.env.completed_cargos / (self.env.current_time / 3600),
            'total_steps': step_count,
            'real_time_elapsed': total_time,
            'simulation_speed': self.env.current_time / total_time if total_time > 0 else 0,
            'hourly_stats': hourly_stats
        }
        
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("评估结果")
        print("=" * 80)
        
        print(f"\n【总体统计】")
        print(f"  仿真时长: {results['duration']:.0f}秒 ({results['duration']/3600:.2f}小时)")
        print(f"  完成货物: {results['completed_cargos']} 件")
        print(f"  超时货物: {results['timed_out_cargos']} 件")
        print(f"  总货物数: {results['total_cargos']} 件")
        print(f"  完成率: {results['completion_rate']*100:.2f}%")
        print(f"  超时率: {results['timeout_rate']*100:.2f}%")
        
        print(f"\n【效率指标】")
        print(f"  平均等待时间: {results['avg_wait_time']:.2f}秒")
        print(f"  每小时吞吐量: {results['throughput_per_hour']:.2f} 件/小时")
        print(f"  总步数: {results['total_steps']}")
        
        print(f"\n【运行性能】")
        print(f"  实际耗时: {results['real_time_elapsed']:.2f}秒 ({results['real_time_elapsed']/60:.2f}分钟)")
        print(f"  仿真加速比: {results['simulation_speed']:.1f}x")
        
        if results['hourly_stats']:
            print(f"\n【每小时统计】")
            print(f"  {'小时':<8} {'完成件数':<12} {'超时件数':<12} {'平均等待(秒)':<15}")
            print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*15}")
            for stat in results['hourly_stats']:
                print(f"  {stat['hour']:<8} {stat['completed']:<12} "
                      f"{stat['timeout']:<12} {stat['avg_wait']:<15.2f}")
        
        print("\n" + "=" * 80)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        result_path = f"eval_results_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果已保存至: {result_path}")
    
    def compare_strategies(self, duration: float = 3600):
        """
        对比不同策略（快速测试用1小时）
        
        Args:
            duration: 测试时长（秒）
        """
        print("=" * 80)
        print("策略对比评估")
        print("=" * 80)
        
        strategies = [
            ('RL决策', True),
            ('规则决策', False)
        ]
        
        comparison = []
        
        for name, use_rl in strategies:
            print(f"\n测试策略: {name}")
            print("-" * 80)
            
            self.env.reset()
            self.high_level_agent.use_rl = use_rl
            
            obs = self.env.reset()
            step_count = 0
            
            while self.env.current_time < duration:
                task_list = self.high_level_controller.compute_action(obs)
                low_actions = self.low_level_controller.compute_actions()
                obs, reward, done = self.env.step(task_list, low_actions)
                step_count += 1
                
                if done:
                    break
            
            result = {
                'strategy': name,
                'completed': self.env.completed_cargos,
                'timeout': self.env.timed_out_cargos,
                'throughput': self.env.completed_cargos / (self.env.current_time / 3600)
            }
            comparison.append(result)
            
            print(f"  完成: {result['completed']} 件")
            print(f"  超时: {result['timeout']} 件")
            print(f"  吞吐量: {result['throughput']:.2f} 件/小时")
        
        print("\n" + "=" * 80)
        print("对比结果")
        print("=" * 80)
        for i, result in enumerate(comparison):
            print(f"{result['strategy']}: "
                  f"完成={result['completed']}, "
                  f"超时={result['timeout']}, "
                  f"吞吐量={result['throughput']:.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AGV系统8小时评估')
    parser.add_argument('--model', type=str, default=None, help='模型路径前缀')
    parser.add_argument('--duration', type=float, default=EPISODE_DURATION_EVAL, 
                       help='评估时长(秒), 默认8小时')
    parser.add_argument('--compare', action='store_true', help='对比不同策略(1小时快速测试)')
    parser.add_argument('--gpu', action='store_true', help='使用GPU')
    
    args = parser.parse_args()
    
    # 创建评估管理器
    manager = EvaluationManager(model_path=args.model, use_gpu=args.gpu)
    
    if args.compare:
        # 策略对比（1小时）
        manager.compare_strategies(duration=3600)
    else:
        # 完整评估
        manager.evaluate_8_hours(duration=args.duration)


if __name__ == "__main__":
    main()
