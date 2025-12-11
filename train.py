"""
训练脚本：训练双智能体系统
"""

import torch
import numpy as np
from typing import Tuple, List
import time
import json
from datetime import datetime
import os

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
        # 高层观测：需要从环境获取实际维度
        sample_high_obs = self.env.get_high_level_observation()
        high_level_obs_dim = len(sample_high_obs)
        high_level_action_dim = (MAX_VEHICLES * NUM_LOADING_STATIONS * 2) + \
                               (MAX_VEHICLES * 2 * NUM_UNLOADING_STATIONS * 2)
        
        # 低层观测：位置(1) + 速度(1) + 其他车辆距离(n-1) + 目标距离(1)
        sample_low_obs = self.env.get_low_level_observation(0)
        low_level_obs_dim = len(sample_low_obs)
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
        
        # 最佳模型追踪
        self.best_avg_reward = float('-inf')
        self.best_avg_completion = 0
        
        # 全局步数统计
        self.total_steps = 0
    
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
        
        # 获取初始观测向量
        high_obs = self.env.get_high_level_observation()
        low_obs = {v_id: self.env.get_low_level_observation(v_id) for v_id in self.env.vehicles.keys()}
        
        # 存储上一步信息
        prev_high_obs = high_obs
        prev_high_action = 0
        prev_low_obs = low_obs.copy()
        prev_low_actions = {v_id: 1 for v_id in self.env.vehicles.keys()}  # 默认保持
        
        while self.env.current_time < EPISODE_DURATION:
            # 高层决策（事件驱动）
            high_level_action = None
            high_action_idx = 0
            
            if self.env.current_time >= next_high_level_decision:
                # 获取高层观测
                high_obs = self.env.get_high_level_observation()
                
                # 选择动作
                high_action_idx = self.high_level_agent.select_action(high_obs)
                high_level_action = self.high_level_controller._decode_action(high_action_idx, obs)
                
                next_high_level_decision = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
            
            # 低层控制（固定时间间隔）
            low_level_actions = {}
            for vehicle_id in self.env.vehicles.keys():
                low_obs[vehicle_id] = self.env.get_low_level_observation(vehicle_id)
                action = self.low_level_agent.select_action(low_obs[vehicle_id])
                low_level_actions[vehicle_id] = action
            
            # 执行一步
            next_obs, reward, done = self.env.step(high_level_action, low_level_actions)
            episode_reward += reward
            step_count += 1
            self.total_steps += 1
            
            # 获取下一步观测
            next_high_obs = self.env.get_high_level_observation()
            next_low_obs = {v_id: self.env.get_low_level_observation(v_id) for v_id in self.env.vehicles.keys()}
            
            # 存储高层经验
            self.high_level_agent.store_transition(
                prev_high_obs, prev_high_action, reward, next_high_obs, done
            )
            
            # 存储低层经验（每个车辆单独存储）
            for vehicle_id in self.env.vehicles.keys():
                self.low_level_agent.store_transition(
                    prev_low_obs[vehicle_id], 
                    prev_low_actions[vehicle_id],
                    reward / len(self.env.vehicles),  # 平均分配奖励
                    next_low_obs[vehicle_id],
                    done
                )
            
            # 更新上一步信息
            prev_high_obs = next_high_obs
            prev_high_action = high_action_idx
            prev_low_obs = next_low_obs.copy()
            prev_low_actions = low_level_actions.copy()
            obs = next_obs
            
            # 定期训练智能体
            if len(self.high_level_agent.memory) >= MIN_REPLAY_SIZE and step_count % TRAIN_FREQUENCY == 0:
                self.high_level_agent.train(batch_size=BATCH_SIZE)
                self.low_level_agent.train(batch_size=BATCH_SIZE)
            
            # 定期更新目标网络
            if step_count % TARGET_UPDATE_FREQUENCY == 0:
                self.high_level_agent.update_target_network()
                self.low_level_agent.update_target_network()
            
            if done:
                break
        
        # 统计
        completed_count = self.env.completed_cargos
        timeout_count = self.env.timed_out_cargos
        avg_wait_time = self.env.total_wait_time / max(1, self.env.cargo_counter)
        
        # 在episode结束后衰减探索率
        self.high_level_agent.decay_epsilon()
        self.low_level_agent.decay_epsilon()
        
        return episode_reward, completed_count, timeout_count, avg_wait_time
    
    def train(self):
        """训练整个系统"""
        print("=" * 80)
        print("开始训练双智能体强化学习系统")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"总episode数: {self.num_episodes}")
        print(f"仿真时长: {EPISODE_DURATION}秒 ({EPISODE_DURATION/3600:.2f}小时)")
        print(f"控制间隔: {LOW_LEVEL_CONTROL_INTERVAL}秒")
        print(f"每episode步数: {MAX_STEPS_PER_EPISODE}")
        print(f"车辆数: {MAX_VEHICLES}")
        print(f"上料口数: {NUM_LOADING_STATIONS}")
        print(f"下料口数: {NUM_UNLOADING_STATIONS}")
        print(f"隐层维度: {HIDDEN_DIM}")
        print(f"学习率: {LEARNING_RATE}")
        print(f"批大小: {BATCH_SIZE}")
        print(f"探索率衰减: {EPSILON_START} -> {EPSILON_END} (衰减系数: {EPSILON_DECAY})")
        print("=" * 80)
        print()
        
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            episode_reward, completed, timeouts, avg_wait = self.train_episode(episode)
            
            episode_time = time.time() - episode_start_time
            
            self.episode_rewards.append(episode_reward)
            self.episode_completions.append(completed)
            self.episode_timeouts.append(timeouts)
            self.episode_avg_wait_times.append(avg_wait)
            self.episode_times.append(episode_time)
            
            # 获取当前探索率和经验池大小
            current_epsilon = self.high_level_agent.epsilon
            replay_size = len(self.high_level_agent.memory)
            
            # 每个episode都打印基本信息
            print(f"Episode {episode+1:4d}/{self.num_episodes} | "
                  f"奖励: {episode_reward:9.2f} | "
                  f"完成: {completed:3d} | "
                  f"超时: {timeouts:2d} | "
                  f"等待: {avg_wait:6.2f}s | "
                  f"ε: {current_epsilon:.3f} | "
                  f"缓冲: {replay_size:6d} | "
                  f"耗时: {episode_time:5.2f}s", flush=True)
            
            # 每10个episode打印统计信息
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_completion = np.mean(self.episode_completions[-10:])
                avg_timeout = np.mean(self.episode_timeouts[-10:])
                avg_wait_10 = np.mean(self.episode_avg_wait_times[-10:])
                
                print(f"  [Episode {episode-8:4d}-{episode+1:4d} 统计] "
                      f"平均奖励: {avg_reward:9.2f} | "
                      f"平均完成: {avg_completion:6.1f} | "
                      f"平均超时: {avg_timeout:4.1f} | "
                      f"平均等待: {avg_wait_10:6.2f}s", flush=True)
                
                # 检查是否是最佳模型
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    self.best_avg_completion = avg_completion
                    self._save_models(prefix="best")
                    print(f"  *** 新最佳模型! 平均奖励: {avg_reward:.2f}, 平均完成: {avg_completion:.1f} ***", flush=True)
                print()
            
            # 定期保存检查点
            if (episode + 1) % SAVE_FREQUENCY == 0:
                self._save_models(prefix=f"checkpoint_ep{episode+1}")
                print(f"  检查点已保存 (Episode {episode+1})", flush=True)
        
        total_time = time.time() - start_time
        print("=" * 80)
        print("训练完成")
        print("=" * 80)
        print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"平均每episode耗时: {total_time/self.num_episodes:.2f}秒")
        print()
        
        # 打印最终统计
        print("最终训练统计:")
        print(f"  总episode数: {self.num_episodes}")
        print(f"  总步数: {self.total_steps}")
        print(f"  平均奖励: {np.mean(self.episode_rewards):.2f} (最后100个: {np.mean(self.episode_rewards[-100:]):.2f})")
        print(f"  最大奖励: {np.max(self.episode_rewards):.2f}")
        print(f"  最小奖励: {np.min(self.episode_rewards):.2f}")
        print(f"  平均完成件数: {np.mean(self.episode_completions):.2f} (最后100个: {np.mean(self.episode_completions[-100:]):.2f})")
        print(f"  平均超时件数: {np.mean(self.episode_timeouts):.2f}")
        print(f"  平均等待时间: {np.mean(self.episode_avg_wait_times):.2f}秒")
        print(f"  最佳平均奖励: {self.best_avg_reward:.2f}")
        print(f"  最佳平均完成: {self.best_avg_completion:.1f}")
        print()
        
        # 保存最终模型
        self._save_models(prefix="final")
    
    def _save_models(self, prefix: str = ""):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            high_level_model_path = f"models/{prefix}_high_level_agent.pt"
            low_level_model_path = f"models/{prefix}_low_level_agent.pt"
            stats_path = f"models/{prefix}_training_stats.json"
        else:
            high_level_model_path = f"models/high_level_agent_{timestamp}.pt"
            low_level_model_path = f"models/low_level_agent_{timestamp}.pt"
            stats_path = f"models/training_stats_{timestamp}.json"
        
        # 保存模型状态字典
        torch.save({
            'q_network': self.high_level_agent.q_network.state_dict(),
            'target_network': self.high_level_agent.target_network.state_dict(),
            'epsilon': self.high_level_agent.epsilon,
        }, high_level_model_path)
        
        torch.save({
            'q_network': self.low_level_agent.q_network.state_dict(),
            'target_network': self.low_level_agent.target_network.state_dict(),
            'epsilon': self.low_level_agent.epsilon,
        }, low_level_model_path)
        
        print(f"模型已保存:")
        print(f"  高层智能体: {high_level_model_path}")
        print(f"  低层智能体: {low_level_model_path}")
        
        # 保存训练统计
        stats = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_completions': [int(x) for x in self.episode_completions],
            'episode_timeouts': [int(x) for x in self.episode_timeouts],
            'episode_avg_wait_times': [float(x) for x in self.episode_avg_wait_times],
            'episode_times': [float(x) for x in self.episode_times],
            'total_steps': self.total_steps,
            'best_avg_reward': float(self.best_avg_reward),
            'best_avg_completion': float(self.best_avg_completion),
            'config': {
                'num_episodes': self.num_episodes,
                'episode_duration': EPISODE_DURATION,
                'control_interval': LOW_LEVEL_CONTROL_INTERVAL,
                'hidden_dim': HIDDEN_DIM,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'gamma': GAMMA,
                'epsilon_start': EPSILON_START,
                'epsilon_end': EPSILON_END,
                'epsilon_decay': EPSILON_DECAY,
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
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
