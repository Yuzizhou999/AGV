"""
训练脚本：训练双智能体系统
支持启发式控制器和RL控制器（PPO）
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import time
import json
from datetime import datetime
import os
import logging

from config import *
from environment import Environment
from heuristic_high_level import HeuristicHighLevelController
from agent_low_level import LowLevelAgent, LowLevelController
from rl_low_level_agent import RLLowLevelController  # SB3的PPO控制器
from custom_ppo_controller import CustomPPOController  # 自定义PPO控制器


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """
    配置日志系统

    Args:
        log_dir: 日志文件存放目录

    Returns:
        配置好的logger实例
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名(带时间戳)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # 创建logger
    logger = logging.getLogger("AGV_Training")
    logger.setLevel(logging.DEBUG)

    # 清除已有的handlers(避免重复配置)
    logger.handlers.clear()

    # 文件handler - 记录所有级别
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台handler - 只记录INFO及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志系统已初始化 | 日志文件: {log_file}")
    logger.info("=" * 80)

    return logger


class TrainingManager:
    """训练管理器"""

    def __init__(self, num_episodes: int = NUM_EPISODES, use_gpu: bool = False,
                 enable_visualization: bool = False, vis_update_interval: int = 10,
                 use_rl_low_level: bool = False, rl_model_path: str = None,
                 use_custom_ppo: bool = True, logger: logging.Logger = None):
        """
        初始化训练管理器

        Args:
            num_episodes: 训练回合数
            use_gpu: 是否使用GPU
            enable_visualization: 是否启用可视化
            vis_update_interval: 可视化更新间隔（每多少步更新一次）
            use_rl_low_level: 是否使用RL底层控制器（PPO）
            rl_model_path: RL模型路径（用于加载已训练模型）
            use_custom_ppo: 是否使用自定义PPO（True）还是SB3的PPO（False）
            logger: 日志记录器实例
        """
        self.num_episodes = num_episodes
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        self.enable_visualization = enable_visualization
        self.vis_update_interval = vis_update_interval
        self.use_rl_low_level = use_rl_low_level
        self.use_custom_ppo = use_custom_ppo
        self.logger = logger if logger is not None else logging.getLogger("AGV_Training")

        # 初始化环境
        self.env = Environment()  # 训练环境：不固定seed
        self.eval_env = Environment()  # 测试环境：独立实例

        # 初始化可视化器（如果启用）
        self.visualizer = None
        if self.enable_visualization:
            try:
                from visualizer import AGVVisualizer
                self.visualizer = AGVVisualizer(self.env)
                self.logger.info("✓ 可视化已启用")
            except ImportError:
                self.logger.warning("⚠ 无法导入可视化模块，禁用可视化")
                self.enable_visualization = False

        # 初始化高层控制器（使用启发式）
        self.high_level_controller = HeuristicHighLevelController(self.env)
        self.eval_high_level_controller = HeuristicHighLevelController(self.eval_env)  # 评估环境专用控制器

        # 初始化底层控制器（根据参数选择）
        if self.use_rl_low_level:
            if self.use_custom_ppo:
                # 使用自定义PPO控制器（推荐）
                self.low_level_controller = CustomPPOController(
                    self.env,
                    model_path=rl_model_path,
                    device=self.device
                )
                self.logger.info("✓ 使用自定义PPO底层控制器（不依赖SB3）")
            else:
                # 使用SB3的PPO控制器
                self.low_level_controller = RLLowLevelController(
                    self.env,
                    model_path=rl_model_path,
                    device=self.device
                )
                self.logger.info("✓ 使用SB3 PPO底层控制器")
        else:
            # 使用原有的DQN控制器
            sample_low_obs = self.env.get_low_level_observation(0)
            low_level_obs_dim = len(sample_low_obs)
            low_level_action_dim = 3
            self.low_level_agent = LowLevelAgent(low_level_obs_dim, low_level_action_dim, self.device)
            self.low_level_controller = LowLevelController(self.low_level_agent, self.env)
            self.logger.info("✓ 使用启发式底层控制器")
        
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
        self.best_eval_reward = float('-inf')  # 最佳测试奖励
        self.episode_actor_losses = []  # Actor loss追踪
        self.episode_critic_losses = []  # Critic loss追踪
        self.episode_entropies = []  # Entropy追踪
        
        # 全局步数统计
        self.total_steps = 0

    def generate_random_seed(self, exclude: int = None) -> int:
        """生成随机seed，排除指定值

        Args:
            exclude: 需要排除的seed值(通常是EVAL_SEED)

        Returns:
            随机生成的seed整数(0-99999)
        """
        while True:
            seed = np.random.randint(0, 100000)
            if exclude is None or seed != exclude:
                return seed

    def aggregate_loss(self, train_stats: dict) -> dict:
        """聚合所有vehicle的loss统计

        Args:
            train_stats: {vehicle_id: {'policy_loss', 'value_loss', 'entropy'}}
                        来自CustomPPOController.update_policies()的返回值

        Returns:
            {'avg_policy_loss': float, 'avg_value_loss': float, 'avg_entropy': float}
        """
        if not train_stats:
            return {
                'avg_policy_loss': 0.0,
                'avg_value_loss': 0.0,
                'avg_entropy': 0.0
            }

        policy_losses = [s['policy_loss'] for s in train_stats.values()]
        value_losses = [s['value_loss'] for s in train_stats.values()]
        entropies = [s['entropy'] for s in train_stats.values()]

        return {
            'avg_policy_loss': np.mean(policy_losses),
            'avg_value_loss': np.mean(value_losses),
            'avg_entropy': np.mean(entropies)
        }

    def evaluate(self, seed: int = None) -> Tuple[float, dict]:
        """评估当前模型性能

        Args:
            seed: 评估使用的随机种子，默认使用EVAL_SEED

        Returns:
            (episode_reward, stats): 测试奖励和统计信息
        """
        if seed is None:
            seed = EVAL_SEED

        # 临时替换low_level_controller的环境引用（评估时使用eval_env）
        original_env = None
        if self.use_rl_low_level:
            original_env = self.low_level_controller.env
            self.low_level_controller.env = self.eval_env

        obs = self.eval_env.reset(seed=seed)
        episode_reward = 0.0
        step_count = 0
        next_high_level_decision = 0.0

        while self.eval_env.current_time < EPISODE_DURATION:
            # 高层决策
            high_level_action = None
            if self.eval_env.current_time >= next_high_level_decision:
                high_level_action = self.eval_high_level_controller.compute_action(obs)
                next_high_level_decision = self.eval_env.current_time + HIGH_LEVEL_DECISION_INTERVAL

            # 低层控制(使用确定性策略)
            if self.use_rl_low_level:
                low_level_actions = self.low_level_controller.compute_actions(deterministic=True)
            else:
                low_level_actions = self.low_level_controller.compute_actions()

            # 执行一步
            next_obs, reward, done = self.eval_env.step(high_level_action, low_level_actions)
            episode_reward += reward
            step_count += 1
            obs = next_obs

            if done:
                break

        # 恢复原环境引用
        if original_env is not None:
            self.low_level_controller.env = original_env

        # 收集统计信息
        stats = {
            'completed': self.eval_env.completed_cargos,
            'total_cargos': self.eval_env.cargo_counter
        }

        return episode_reward, stats

    def train_episode(self, episode_idx: int, seed: int = None) -> Tuple[float, int, int, int, float, int, int, int, dict]:
        """训练一个episode

        Args:
            episode_idx: Episode索引
            seed: 环境重置使用的随机种子，None时使用随机值

        Returns:
            (total_reward, completed_count, completed_timeout_count,
             waiting_cargos_normal, waiting_cargos_timeout,
             avg_wait_time, avg_completion_time,
             total_cargos, waiting_cargos, on_vehicle_cargos, train_stats)
        """
        obs = self.env.reset(seed=seed)
        episode_reward = 0.0
        step_count = 0
        
        # 高层决策时间管理
        next_high_level_decision = 0.0
        
        while self.env.current_time < EPISODE_DURATION:
            # 高层决策（事件驱动）
            high_level_action = None
            
            if self.env.current_time >= next_high_level_decision:
                # 使用启发式控制器计算高层动作
                high_level_action = self.high_level_controller.compute_action(obs)
                next_high_level_decision = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
            
            # 低层控制
            if self.use_rl_low_level:
                # 使用RL控制器（PPO）- 返回连续动作
                # 自定义PPO会自动收集经验
                low_level_actions = self.low_level_controller.compute_actions(deterministic=False)
            else:
                # 使用启发式控制器 - 返回离散动作
                low_level_actions = self.low_level_controller.compute_actions()
            
            # 执行一步
            next_obs, reward, done = self.env.step(high_level_action, low_level_actions)
            episode_reward += reward
            step_count += 1
            self.total_steps += 1
            
            # 如果使用自定义PPO，存储转移
            if self.use_rl_low_level and self.use_custom_ppo:
                self.low_level_controller.store_transitions(done=done)
            
            # 更新上一步信息
            obs = next_obs
            
            # 更新可视化（如果启用）
            if self.enable_visualization and step_count % self.vis_update_interval == 0:
                self.visualizer.update()
            
            if done:
                break
        
        # Episode结束后，训练PPO模型
        train_stats = None
        if self.use_rl_low_level and self.use_custom_ppo:
            # 使用自定义PPO，直接从缓冲区更新
            train_stats = self.low_level_controller.update_policies()
            # 可以打印训练统计（可选）
            # print(f"  训练统计: {train_stats}")
        elif self.use_rl_low_level and not self.use_custom_ppo:
            # 使用SB3的PPO，需要通过环境交互训练
            # 这部分逻辑保留用于兼容性
            pass
        
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
        
        # 启发式高层控制器不需要epsilon衰减
        # self.low_level_agent.decay_epsilon()  # 使用启发式控制，不需要探索

        return (episode_reward, completed_count, completed_timeout_count,
                waiting_cargos_normal, waiting_cargos_timeout,
                avg_wait_time, avg_completion_time,
                total_cargos, waiting_cargos, on_vehicle_cargos, train_stats)
    
    def train(self):
        """训练整个系统"""
        self.logger.info("=" * 80)
        self.logger.info("开始运行启发式调度系统评估")
        self.logger.info("=" * 80)
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"总episode数: {self.num_episodes}")
        self.logger.info(f"仿真时长: {EPISODE_DURATION}秒 ({EPISODE_DURATION/3600:.2f}小时)")
        self.logger.info(f"控制间隔: {LOW_LEVEL_CONTROL_INTERVAL}秒")
        self.logger.info(f"每episode步数: {MAX_STEPS_PER_EPISODE}")
        self.logger.info(f"车辆数: {MAX_VEHICLES}")
        self.logger.info(f"上料口数: {NUM_LOADING_STATIONS}")
        self.logger.info(f"下料口数: {NUM_UNLOADING_STATIONS}")
        self.logger.info(f"高层控制: 启发式规则（最近距离分配）")
        self.logger.info(f"底层控制: {'自定义PPO' if self.use_custom_ppo else 'SB3 PPO' if self.use_rl_low_level else '启发式控制器'}")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        start_time = time.time()
        os.makedirs("models", exist_ok=True)
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()

            # 生成随机seed(排除EVAL_SEED)
            train_seed = self.generate_random_seed(exclude=EVAL_SEED)

            # 训练一个episode
            (episode_reward, completed, completed_timeout,
             waiting_normal, waiting_timeout,
             avg_wait, avg_completion,
             total_cargos, waiting_cargos, on_vehicle_cargos, train_stats) = self.train_episode(episode, train_seed)

            episode_time = time.time() - episode_start_time

            self.episode_rewards.append(episode_reward)
            self.episode_completions.append(completed)
            self.episode_completed_timeouts.append(completed_timeout)
            self.episode_waiting_normals.append(waiting_normal)
            self.episode_waiting_timeouts.append(waiting_timeout)
            self.episode_avg_wait_times.append(avg_wait)
            self.episode_avg_completion_times.append(avg_completion)
            self.episode_times.append(episode_time)

            # 记录loss(仅在使用RL时)
            if train_stats and self.use_rl_low_level:
                avg_loss = self.aggregate_loss(train_stats)
                self.episode_actor_losses.append(avg_loss['avg_policy_loss'])
                self.episode_critic_losses.append(avg_loss['avg_value_loss'])
                self.episode_entropies.append(avg_loss['avg_entropy'])

            # 计算正常完成数量
            completed_normal = completed - completed_timeout

            # 每个episode都打印基本信息
            self.logger.info(f"Episode {episode+1:4d}/{self.num_episodes} (seed={train_seed}) | "
                  f"奖励: {episode_reward:9.2f} | "
                  f"完成: {completed:3d} (正常: {completed_normal:3d}, 超时: {completed_timeout:2d}) | "
                  f"待取: {waiting_cargos:2d} (正常: {waiting_normal:2d}, 超时: {waiting_timeout:2d}) | "
                  f"总货: {total_cargos:3d} | "
                  f"在车: {on_vehicle_cargos:2d} | "
                  f"等待: {avg_wait:6.2f}s | "
                  f"完成: {avg_completion:6.2f}s | "
                  f"耗时: {episode_time:5.2f}s")

            # 打印loss(仅在使用RL时)
            if train_stats and self.use_rl_low_level:
                avg_loss = self.aggregate_loss(train_stats)
                self.logger.info(f"  [训练] Loss: actor={avg_loss['avg_policy_loss']:.4f}, "
                                f"critic={avg_loss['avg_value_loss']:.4f}, "
                                f"entropy={avg_loss['avg_entropy']:.4f}")

            # 每EVAL_INTERVAL个episode评估
            if (episode + 1) % EVAL_INTERVAL == 0:
                eval_reward, eval_stats = self.evaluate()

                # 打印评估结果
                self.logger.info(f"  [评估] (seed={EVAL_SEED}) 测试奖励: {eval_reward:.2f} | "
                                f"测试完成: {eval_stats['completed']}")

                # 检查是否是最佳模型(使用测试奖励)
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.logger.info(f"  *** 新最佳测试性能! 测试奖励: {eval_reward:.2f} ***")

                    # 保存最佳模型
                    if self.use_rl_low_level:
                        self.low_level_controller.save_models("models", prefix="rl_low_level_best")

                # 保持原有的10个episode统计(基于训练奖励)
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_completion = np.mean(self.episode_completions[-10:])
                avg_completed_timeout = np.mean(self.episode_completed_timeouts[-10:])
                avg_waiting_normal = np.mean(self.episode_waiting_normals[-10:])
                avg_waiting_timeout = np.mean(self.episode_waiting_timeouts[-10:])
                avg_wait_10 = np.mean(self.episode_avg_wait_times[-10:])
                avg_completion_time_10 = np.mean(self.episode_avg_completion_times[-10:])

                self.logger.info(f"  [Episode {episode-8:4d}-{episode+1:4d} 统计] "
                      f"平均奖励: {avg_reward:9.2f} | "
                      f"平均完成: {avg_completion:6.1f} (正常: {avg_completion-avg_completed_timeout:5.1f}, 超时: {avg_completed_timeout:4.1f}) | "
                      f"平均待取: {avg_waiting_normal+avg_waiting_timeout:4.1f} (正常: {avg_waiting_normal:4.1f}, 超时: {avg_waiting_timeout:4.1f}) | "
                      f"平均等待: {avg_wait_10:6.2f}s | "
                      f"平均完成: {avg_completion_time_10:6.2f}s")

                self.logger.info("")
            
        total_time = time.time() - start_time
        self.logger.info("=" * 80)
        self.logger.info("评估完成")
        self.logger.info("=" * 80)
        self.logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        self.logger.info(f"平均每episode耗时: {total_time/self.num_episodes:.2f}秒")
        self.logger.info("")

        # 打印最终统计
        self.logger.info("最终评估统计:")
        self.logger.info(f"  总episode数: {self.num_episodes}")
        self.logger.info(f"  平均奖励: {np.mean(self.episode_rewards):.2f} (最后100个: {np.mean(self.episode_rewards[-100:]):.2f})")
        self.logger.info(f"  最大奖励: {np.max(self.episode_rewards):.2f}")
        self.logger.info(f"  最小奖励: {np.min(self.episode_rewards):.2f}")
        
        # 如果启用了可视化，显示训练统计图表
        if self.enable_visualization and self.visualizer is not None:
            self.logger.info("\n正在生成可视化统计图表...")
            self.visualizer.plot_statistics(save_path="training_visualization_stats.png")
            self.visualizer.close()

        avg_completed = np.mean(self.episode_completions)
        avg_completed_timeout = np.mean(self.episode_completed_timeouts)
        avg_completed_normal = avg_completed - avg_completed_timeout
        avg_waiting_normal = np.mean(self.episode_waiting_normals)
        avg_waiting_timeout = np.mean(self.episode_waiting_timeouts)

        self.logger.info(f"  平均完成件数: {avg_completed:.2f} (最后100个: {np.mean(self.episode_completions[-100:]):.2f})")
        self.logger.info(f"    - 正常完成: {avg_completed_normal:.2f}")
        self.logger.info(f"    - 超时完成: {avg_completed_timeout:.2f}")
        self.logger.info(f"  平均待取件数: {avg_waiting_normal + avg_waiting_timeout:.2f}")
        self.logger.info(f"    - 正常等待: {avg_waiting_normal:.2f}")
        self.logger.info(f"    - 超时等待: {avg_waiting_timeout:.2f}")
        self.logger.info(f"  平均等待时间(到被取走): {np.mean(self.episode_avg_wait_times):.2f}秒")
        self.logger.info(f"  平均完成时间(到完成下料): {np.mean(self.episode_avg_completion_times):.2f}秒")
        self.logger.info(f"  最佳平均奖励: {self.best_avg_reward:.2f}")
        self.logger.info(f"  最佳平均完成: {self.best_avg_completion:.1f}")

        # 新增：最佳评估奖励
        self.logger.info(f"  最佳测试奖励: {self.best_eval_reward:.2f}")

        # 新增：loss统计(仅在使用RL时)
        if self.use_rl_low_level and self.episode_actor_losses:
            self.logger.info(f"  平均Actor Loss: {np.mean(self.episode_actor_losses):.4f}")
            self.logger.info(f"  平均Critic Loss: {np.mean(self.episode_critic_losses):.4f}")
            self.logger.info(f"  平均Entropy: {np.mean(self.episode_entropies):.4f}")

        self.logger.info("")
        
        # 保存评估统计
        self._save_stats()
    
    def _save_stats(self):
        """保存评估统计数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = f"heuristic_evaluation_stats_{timestamp}.json"
        
        # 保存评估统计
        stats = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_completions': [int(x) for x in self.episode_completions],
            'episode_completed_timeouts': [int(x) for x in self.episode_completed_timeouts],
            'episode_waiting_normals': [int(x) for x in self.episode_waiting_normals],
            'episode_waiting_timeouts': [int(x) for x in self.episode_waiting_timeouts],
            'episode_avg_wait_times': [float(x) for x in self.episode_avg_wait_times],
            'episode_avg_completion_times': [float(x) for x in self.episode_avg_completion_times],
            'episode_times': [float(x) for x in self.episode_times],
            'best_avg_reward': float(self.best_avg_reward),
            'best_avg_completion': float(self.best_avg_completion),

            # 新增字段
            'best_eval_reward': float(self.best_eval_reward),
            'episode_actor_losses': [float(x) for x in self.episode_actor_losses],
            'episode_critic_losses': [float(x) for x in self.episode_critic_losses],
            'episode_entropies': [float(x) for x in self.episode_entropies],

            'config': {
                'num_episodes': self.num_episodes,
                'episode_duration': EPISODE_DURATION,
                'control_interval': LOW_LEVEL_CONTROL_INTERVAL,
                'high_level_control': 'heuristic_nearest_distance',
                'low_level_control': 'heuristic_controller',
            }
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.logger.info(f"评估统计已保存: {stats_path}")



def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练AGV调度系统')
    parser.add_argument('--visualize', action='store_true',
                       help='启用可视化（会显著降低训练速度）')
    parser.add_argument('--vis-interval', type=int, default=50,
                       help='可视化更新间隔（步数）')
    parser.add_argument('--use-rl-low-level', action='store_false',
                       help='使用RL底层控制器（PPO）代替启发式控制器')
    parser.add_argument('--use-sb3-ppo', action='store_true',
                       help='使用Stable-Baselines3的PPO（默认使用自定义PPO）')
    parser.add_argument('--rl-model-path', type=str, default=None,
                       help='RL模型路径（用于继续训练或评估）')
    args = parser.parse_args()

    # 初始化日志系统
    logger = setup_logger()

    # 检查GPU可用性
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU可用: {use_gpu}")
    if use_gpu:
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")

    if args.visualize:
        logger.warning("⚠ 可视化已启用 - 训练速度会显著降低")
        logger.info(f"  可视化更新间隔: {args.vis_interval} 步")

    # 打印控制器配置
    if args.use_rl_low_level:
        if args.use_sb3_ppo:
            logger.info("✓ 底层控制: RL智能体（Stable-Baselines3 PPO）")
        else:
            logger.info("✓ 底层控制: RL智能体（自定义PPO - 推荐）")
        if args.rl_model_path:
            logger.info(f"  加载模型: {args.rl_model_path}")
    else:
        logger.info("✓ 底层控制: 启发式控制器")

    logger.info("")

    # 创建训练管理器
    manager = TrainingManager(
        num_episodes=NUM_EPISODES,
        use_gpu=use_gpu,
        enable_visualization=False,
        vis_update_interval=args.vis_interval,
        use_rl_low_level=args.use_rl_low_level,
        rl_model_path=args.rl_model_path,
        use_custom_ppo=not args.use_sb3_ppo,  # 默认使用自定义PPO
        logger=logger
    )

    # 开始训练
    manager.train()


if __name__ == "__main__":
    main()
