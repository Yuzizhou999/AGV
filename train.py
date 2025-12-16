"""
训练脚本：训练双智能体系统
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
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
    
    def __init__(self, num_episodes: int = NUM_EPISODES, use_gpu: bool = False, 
                 enable_visualization: bool = False, vis_update_interval: int = 10):
        """
        初始化训练管理器
        
        Args:
            num_episodes: 训练回合数
            use_gpu: 是否使用GPU
            enable_visualization: 是否启用可视化
            vis_update_interval: 可视化更新间隔（每多少步更新一次）
        """
        self.num_episodes = num_episodes
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        self.enable_visualization = enable_visualization
        self.vis_update_interval = vis_update_interval
        
        # 初始化环境
        self.env = Environment(seed=42)
        
        # 初始化可视化器（如果启用）
        self.visualizer = None
        if self.enable_visualization:
            try:
                from visualizer import AGVVisualizer
                self.visualizer = AGVVisualizer(self.env)
                print("✓ 可视化已启用")
            except ImportError:
                print("⚠ 无法导入可视化模块，禁用可视化")
                self.enable_visualization = False
        
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
        self.episode_completed_timeouts = []
        self.episode_waiting_normals = []  # 不超时等待
        self.episode_waiting_timeouts = []  # 超时等待
        self.episode_avg_wait_times = []  # 平均等待时间(到被取走)
        self.episode_avg_completion_times = []  # 平均完成时间(到完成下料)
        self.episode_times = []
        
        # 最佳模型追踪
        self.best_avg_reward = float('-inf')
        self.best_avg_completion = 0
        
        # 全局步数统计
        self.total_steps = 0
    
    def train_episode(self, episode_idx: int) -> Tuple[float, int, int, int, float, int, int, int]:
        """
        训练一个episode
        
        Returns:
            (total_reward, completed_count, completed_timeout_count, 
             waiting_cargos_normal, waiting_cargos_timeout,
             avg_wait_time, avg_completion_time,
             total_cargos, waiting_cargos, on_vehicle_cargos)
        """
        obs = self.env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # 高层决策时间管理
        next_high_level_decision = 0.0
        
        # 获取初始观测向量
        high_obs = self.env.get_high_level_observation()
        # 低层使用启发式控制，不需要观测
        # low_obs = {v_id: self.env.get_low_level_observation(v_id) for v_id in self.env.vehicles.keys()}
        
        # 存储上一步信息
        prev_high_obs = high_obs
        prev_high_action = 0
        # 低层使用启发式控制，不需要存储
        # prev_low_obs = low_obs.copy()
        # prev_low_actions = {v_id: 1 for v_id in self.env.vehicles.keys()}  # 默认保持
        
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
            
            # 低层控制（使用启发式控制器替代神经网络）
            # low_level_actions = {}
            # for vehicle_id in self.env.vehicles.keys():
            #     low_obs[vehicle_id] = self.env.get_low_level_observation(vehicle_id)
            #     action = self.low_level_agent.select_action(low_obs[vehicle_id])
            #     low_level_actions[vehicle_id] = action
            
            # 执行一步（使用启发式控制器，low_level_actions=None）
            next_obs, reward, done = self.env.step(high_level_action, low_level_actions=None)
            episode_reward += reward
            step_count += 1
            self.total_steps += 1
            
            # 获取下一步观测（只需要高层观测）
            next_high_obs = self.env.get_high_level_observation()
            # next_low_obs = {v_id: self.env.get_low_level_observation(v_id) for v_id in self.env.vehicles.keys()}  # 不需要
            
            # 存储高层经验
            self.high_level_agent.store_transition(
                prev_high_obs, prev_high_action, reward, next_high_obs, done
            )
            
            # 低层使用启发式控制，不需要训练
            # （注释掉低层智能体的经验存储）
            # for vehicle_id in self.env.vehicles.keys():
            #     self.low_level_agent.store_transition(
            #         prev_low_obs[vehicle_id], 
            #         prev_low_actions[vehicle_id],
            #         reward / len(self.env.vehicles),
            #         next_low_obs[vehicle_id],
            #         done
            #     )
            
            # 更新上一步信息
            prev_high_obs = next_high_obs
            prev_high_action = high_action_idx
            # prev_low_obs = next_low_obs.copy()  # 不再需要
            # prev_low_actions = low_level_actions.copy()  # 不再需要
            obs = next_obs
            
            # 定期训练智能体（只训练高层智能体）
            if len(self.high_level_agent.memory) >= MIN_REPLAY_SIZE and step_count % TRAIN_FREQUENCY == 0:
                self.high_level_agent.train(batch_size=BATCH_SIZE)
                # self.low_level_agent.train(batch_size=BATCH_SIZE)  # 使用启发式控制，不需要训练
            
            # 定期更新目标网络（只更新高层）
            if step_count % TARGET_UPDATE_FREQUENCY == 0:
                self.high_level_agent.update_target_network()
                # self.low_level_agent.update_target_network()  # 使用启发式控制，不需要训练
            
            # 更新可视化（如果启用）
            if self.enable_visualization and step_count % self.vis_update_interval == 0:
                self.visualizer.update()
            
            if done:
                break
        
        # 统计
        completed_count = self.env.completed_cargos
        # 统计完成货物中超时的数量(从completed_cargo_list统计,因为完成的货物已从cargos中删除)
        completed_timeout_count = sum(1 for c in self.env.completed_cargo_list
                                     if c.get('wait_time', 0) > CARGO_TIMEOUT)
        # 统计等待中的货物:分为超时等待和不超时等待
        waiting_cargos_normal = sum(1 for c in self.env.cargos.values() 
                                   if c.completion_time is None 
                                   and c.current_location.startswith("IP_")
                                   and not c.is_timeout(self.env.current_time) and c.picked_up_time is None)
        waiting_cargos_timeout = sum(1 for c in self.env.cargos.values() 
                                    if c.completion_time is None 
                                    and c.current_location.startswith("IP_")
                                    and c.is_timeout(self.env.current_time) and c.picked_up_time is None)
        
        total_cargos = self.env.cargo_counter
        waiting_cargos = waiting_cargos_normal + waiting_cargos_timeout  # 总等待数
        on_vehicle_cargos = sum(1 for c in self.env.cargos.values()
                               if c.completion_time is None and c.current_location.startswith("vehicle_"))
        
        # 平均等待时间(从到达到被取走)
        avg_wait_time = self.env.total_wait_time / max(1, self.env.completed_cargos)
        
        # 平均完成时间(从到达到完成下料)
        if self.env.completed_cargo_list:
            avg_completion_time = sum(c['completion_time'] - c['arrival_time'] 
                                     for c in self.env.completed_cargo_list) / len(self.env.completed_cargo_list)
        else:
            avg_completion_time = 0.0
        
        # 在episode结束后衰减探索率（只针对高层智能体）
        self.high_level_agent.decay_epsilon()
        # self.low_level_agent.decay_epsilon()  # 使用启发式控制，不需要探索
        
        return (episode_reward, completed_count, completed_timeout_count, 
                waiting_cargos_normal, waiting_cargos_timeout, 
                avg_wait_time, avg_completion_time,
                total_cargos, waiting_cargos, on_vehicle_cargos)
    
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
        print(f"底层控制: 启发式控制器 (不使用神经网络)")
        print("=" * 80)
        print()
        
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            (episode_reward, completed, completed_timeout, 
             waiting_normal, waiting_timeout, 
             avg_wait, avg_completion,
             total_cargos, waiting_cargos, on_vehicle_cargos) = self.train_episode(episode)
            
            episode_time = time.time() - episode_start_time
            
            self.episode_rewards.append(episode_reward)
            self.episode_completions.append(completed)
            self.episode_completed_timeouts.append(completed_timeout)
            self.episode_waiting_normals.append(waiting_normal)
            self.episode_waiting_timeouts.append(waiting_timeout)
            self.episode_avg_wait_times.append(avg_wait)
            self.episode_avg_completion_times.append(avg_completion)
            self.episode_times.append(episode_time)
            
            # 获取当前探索率和经验池大小
            current_epsilon = self.high_level_agent.epsilon
            replay_size = len(self.high_level_agent.memory)
            
            # 计算正常完成数量
            completed_normal = completed - completed_timeout
            
            # 每个episode都打印基本信息
            print(f"Episode {episode+1:4d}/{self.num_episodes} | "
                  f"奖励: {episode_reward:9.2f} | "
                  f"完成: {completed:3d} (正常: {completed_normal:3d}, 超时: {completed_timeout:2d}) | "
                  f"待取: {waiting_cargos:2d} (正常: {waiting_normal:2d}, 超时: {waiting_timeout:2d}) | "
                  f"总货: {total_cargos:3d} | "
                  f"在车: {on_vehicle_cargos:2d} | "
                  f"等待: {avg_wait:6.2f}s | "
                  f"完成: {avg_completion:6.2f}s | "
                  f"ε: {current_epsilon:.3f} | "
                  f"缓冲: {replay_size:6d} | "
                  f"耗时: {episode_time:5.2f}s", flush=True)
            
            # 每10个episode打印统计信息
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_completion = np.mean(self.episode_completions[-10:])
                avg_completed_timeout = np.mean(self.episode_completed_timeouts[-10:])
                avg_waiting_normal = np.mean(self.episode_waiting_normals[-10:])
                avg_waiting_timeout = np.mean(self.episode_waiting_timeouts[-10:])
                avg_wait_10 = np.mean(self.episode_avg_wait_times[-10:])
                avg_completion_time_10 = np.mean(self.episode_avg_completion_times[-10:])
                
                print(f"  [Episode {episode-8:4d}-{episode+1:4d} 统计] "
                      f"平均奖励: {avg_reward:9.2f} | "
                      f"平均完成: {avg_completion:6.1f} (正常: {avg_completion-avg_completed_timeout:5.1f}, 超时: {avg_completed_timeout:4.1f}) | "
                      f"平均待取: {avg_waiting_normal+avg_waiting_timeout:4.1f} (正常: {avg_waiting_normal:4.1f}, 超时: {avg_waiting_timeout:4.1f}) | "
                      f"平均等待: {avg_wait_10:6.2f}s | "
                      f"平均完成: {avg_completion_time_10:6.2f}s", flush=True)
                
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
        
        # 如果启用了可视化，显示训练统计图表
        if self.enable_visualization and self.visualizer is not None:
            print("\n正在生成可视化统计图表...")
            self.visualizer.plot_statistics(save_path="training_visualization_stats.png")
            self.visualizer.close()
        
        avg_completed = np.mean(self.episode_completions)
        avg_completed_timeout = np.mean(self.episode_completed_timeouts)
        avg_completed_normal = avg_completed - avg_completed_timeout
        avg_waiting_normal = np.mean(self.episode_waiting_normals)
        avg_waiting_timeout = np.mean(self.episode_waiting_timeouts)
        
        print(f"  平均完成件数: {avg_completed:.2f} (最后100个: {np.mean(self.episode_completions[-100:]):.2f})")
        print(f"    - 正常完成: {avg_completed_normal:.2f}")
        print(f"    - 超时完成: {avg_completed_timeout:.2f}")
        print(f"  平均待取件数: {avg_waiting_normal + avg_waiting_timeout:.2f}")
        print(f"    - 正常等待: {avg_waiting_normal:.2f}")
        print(f"    - 超时等待: {avg_waiting_timeout:.2f}")
        print(f"  平均等待时间(到被取走): {np.mean(self.episode_avg_wait_times):.2f}秒")
        print(f"  平均完成时间(到完成下料): {np.mean(self.episode_avg_completion_times):.2f}秒")
        print(f"  最佳平均奖励: {self.best_avg_reward:.2f}")
        print(f"  最佳平均完成: {self.best_avg_completion:.1f}")
        print()
        
        # 保存最终模型
        self._save_models(prefix="final")
    
    def _save_models(self, prefix: str = ""):
        """保存训练好的模型（只保存高层智能体）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            high_level_model_path = f"models/{prefix}_high_level_agent.pt"
            # low_level_model_path = f"models/{prefix}_low_level_agent.pt"  # 使用启发式控制，不需要保存
            stats_path = f"models/{prefix}_training_stats.json"
        else:
            high_level_model_path = f"models/high_level_agent_{timestamp}.pt"
            # low_level_model_path = f"models/low_level_agent_{timestamp}.pt"  # 使用启发式控制，不需要保存
            stats_path = f"models/training_stats_{timestamp}.json"
        
        # 保存模型状态字典（只保存高层）
        torch.save({
            'q_network': self.high_level_agent.q_network.state_dict(),
            'target_network': self.high_level_agent.target_network.state_dict(),
            'epsilon': self.high_level_agent.epsilon,
        }, high_level_model_path)
        
        # 低层使用启发式控制，不需要保存模型
        # torch.save({
        #     'q_network': self.low_level_agent.q_network.state_dict(),
        #     'target_network': self.low_level_agent.target_network.state_dict(),
        #     'epsilon': self.low_level_agent.epsilon,
        # }, low_level_model_path)
        
        print(f"模型已保存:")
        print(f"  高层智能体: {high_level_model_path}")
        # print(f"  低层智能体: {low_level_model_path}")  # 使用启发式控制
        print(f"  (低层使用启发式控制，无需保存模型)")
        
        # 保存训练统计
        stats = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_completions': [int(x) for x in self.episode_completions],
            'episode_completed_timeouts': [int(x) for x in self.episode_completed_timeouts],
            'episode_waiting_normals': [int(x) for x in self.episode_waiting_normals],
            'episode_waiting_timeouts': [int(x) for x in self.episode_waiting_timeouts],
            'episode_avg_wait_times': [float(x) for x in self.episode_avg_wait_times],
            'episode_avg_completion_times': [float(x) for x in self.episode_avg_completion_times],
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
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练AGV调度系统')
    parser.add_argument('--visualize', action='store_true', 
                       help='启用可视化（会显著降低训练速度）')
    parser.add_argument('--vis-interval', type=int, default=50,
                       help='可视化更新间隔（步数）')
    args = parser.parse_args()
    
    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    if args.visualize:
        print("⚠ 可视化已启用 - 训练速度会显著降低")
        print(f"  可视化更新间隔: {args.vis_interval} 步")
    print()
    
    # 创建训练管理器
    manager = TrainingManager(
        num_episodes=NUM_EPISODES, 
        use_gpu=use_gpu,
        enable_visualization=False,
        vis_update_interval=args.vis_interval
    )
    
    # 开始训练
    manager.train()


if __name__ == "__main__":
    main()
