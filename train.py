"""
训练脚本：训练双智能体系统
"""

import torch
import numpy as np
from typing import Tuple, List
import time
import json
from datetime import datetime

from config import *
from environment import Environment
from agent_high_level import HighLevelAgent, HighLevelController
from agent_low_level import LowLevelAgent, LowLevelController


class TrainingManager:
    """训练管理器"""
    
    def __init__(self, num_episodes: int = NUM_EPISODES, use_gpu: bool = False):
        self.num_episodes = num_episodes
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        
        # 初始化环境
        self.env = Environment(seed=42)
        
        # 计算观测维度
        # 高层观测：2台车(4) + 2个上料口(3) + 全局(2) = 9
        high_level_obs_dim = MAX_VEHICLES * 4 + NUM_LOADING_STATIONS * 3 + 2
        high_level_action_dim = (MAX_VEHICLES * NUM_LOADING_STATIONS * 2) + \
                               (MAX_VEHICLES * 2 * NUM_UNLOADING_STATIONS * 2)
        
        # 低层观测：位置(1) + 速度(1) + 前车距离(n-1) + 目标距离(1) = 3 + n
        low_level_obs_dim = 3 + MAX_VEHICLES
        low_level_action_dim = 3
        
        # 初始化智能体
        self.high_level_agent = HighLevelAgent(high_level_obs_dim, high_level_action_dim, self.device)
        self.low_level_agent = LowLevelAgent(low_level_obs_dim, low_level_action_dim, self.device)
        
        # 初始化控制器
        self.high_level_controller = HighLevelController(self.high_level_agent, self.env)
        self.low_level_controller = LowLevelController(self.low_level_agent, self.env)
        
        # 统计信息
        self.episode_rewards = []
        self.episode_completions = []
        self.episode_timeouts = []
        self.episode_avg_wait_times = []
        self.episode_times = []
    
    def train_episode(self, episode_idx: int) -> Tuple[float, int, int, float]:
        """
        训练一个episode
        
        Returns:
            (total_reward, completed_count, timeout_count, avg_wait_time)
        """
        obs = self.env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # 高层决策时间管理
        next_high_level_decision = 0.0
        
        while self.env.current_time < EPISODE_DURATION:
            # 高层决策（事件驱动）
            high_level_action = None
            if self.env.current_time >= next_high_level_decision:
                high_level_action = self.high_level_controller.compute_action(obs)
                next_high_level_decision = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
            
            # 低层控制（固定时间间隔）
            low_level_actions = self.low_level_controller.compute_actions()
            
            # 执行一步
            obs, reward, done = self.env.step(high_level_action, low_level_actions)
            episode_reward += reward
            step_count += 1
            
            # 训练智能体（每100步训练一次以加速）
            if step_count % 100 == 0:
                self.high_level_agent.train(batch_size=BATCH_SIZE)
                self.low_level_agent.train(batch_size=BATCH_SIZE)
            
            # 打印进度（每个episode内部）
            if step_count % 200 == 0:
                progress = self.env.current_time / EPISODE_DURATION * 100
                print(f"    Episode进度: {progress:.1f}% | 步数: {step_count} | 货物: {len(self.env.cargos)} | 完成: {self.env.completed_cargos}", flush=True)
            
            if done:
                break
        
        # 统计
        completed_count = self.env.completed_cargos
        timeout_count = self.env.timed_out_cargos
        avg_wait_time = self.env.total_wait_time / max(1, self.env.cargo_counter)
        
        return episode_reward, completed_count, timeout_count, avg_wait_time
    
    def train(self):
        """训练整个系统"""
        print("=" * 80)
        print("开始训练双智能体强化学习系统")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"总episode数: {self.num_episodes}")
        print(f"仿真时长: {EPISODE_DURATION}秒 ({EPISODE_DURATION/3600:.1f}小时)")
        print(f"车辆数: {MAX_VEHICLES}")
        print(f"上料口数: {NUM_LOADING_STATIONS}")
        print(f"下料口数: {NUM_UNLOADING_STATIONS}")
        print("=" * 80)
        print()
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            episode_reward, completed, timeouts, avg_wait = self.train_episode(episode)
            
            episode_time = time.time() - episode_start_time
            
            self.episode_rewards.append(episode_reward)
            self.episode_completions.append(completed)
            self.episode_timeouts.append(timeouts)
            self.episode_avg_wait_times.append(avg_wait)
            self.episode_times.append(episode_time)
            
            # 定期打印信息（每个episode都打印）
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) > 0 else episode_reward
            avg_completion = np.mean(self.episode_completions[-10:]) if len(self.episode_completions) > 0 else completed
            avg_timeout = np.mean(self.episode_timeouts[-10:]) if len(self.episode_timeouts) > 0 else timeouts
            
            print(f"Episode {episode+1:3d}/{self.num_episodes} | "
                  f"总奖励: {episode_reward:8.2f} | "
                  f"完成件数: {completed:3d} | "
                  f"超时件数: {timeouts:2d} | "
                  f"平均等待时间: {avg_wait:6.2f}s | "
                  f"耗时: {episode_time:6.2f}s", flush=True)
            
            if (episode + 1) % 10 == 0:
                print(f"  [最近10个Episode统计] "
                      f"平均奖励: {avg_reward:8.2f} | "
                      f"平均完成件数: {avg_completion:6.1f} | "
                      f"平均超时: {avg_timeout:4.1f}", flush=True)
                print()
        
        total_time = time.time() - start_time
        print("=" * 80)
        print("训练完成")
        print("=" * 80)
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"平均每episode耗时: {total_time/self.num_episodes:.2f}秒")
        print()
        
        # 打印最终统计
        print("最终统计信息:")
        print(f"  平均奖励: {np.mean(self.episode_rewards):.2f}")
        print(f"  最大奖励: {np.max(self.episode_rewards):.2f}")
        print(f"  最小奖励: {np.min(self.episode_rewards):.2f}")
        print(f"  平均完成件数: {np.mean(self.episode_completions):.2f}")
        print(f"  平均超时件数: {np.mean(self.episode_timeouts):.2f}")
        print(f"  平均等待时间: {np.mean(self.episode_avg_wait_times):.2f}秒")
        print()
        
        # 保存模型
        self._save_models()
    
    def _save_models(self):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        high_level_model_path = f"models/high_level_agent_{timestamp}.pt"
        low_level_model_path = f"models/low_level_agent_{timestamp}.pt"
        
        import os
        os.makedirs("models", exist_ok=True)
        
        torch.save(self.high_level_agent.q_network.state_dict(), high_level_model_path)
        torch.save(self.low_level_agent.q_network.state_dict(), low_level_model_path)
        
        print(f"模型已保存:")
        print(f"  高层智能体: {high_level_model_path}")
        print(f"  低层智能体: {low_level_model_path}")
        
        # 保存训练统计
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_completions': self.episode_completions,
            'episode_timeouts': self.episode_timeouts,
            'episode_avg_wait_times': self.episode_avg_wait_times,
            'episode_times': self.episode_times
        }
        
        stats_path = f"models/training_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  训练统计: {stats_path}")


def main():
    """主函数"""
    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()
    
    # 创建训练管理器
    manager = TrainingManager(num_episodes=NUM_EPISODES, use_gpu=use_gpu)
    
    # 开始训练
    manager.train()


if __name__ == "__main__":
    main()
