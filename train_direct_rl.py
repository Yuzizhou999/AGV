"""
直接控制RL训练脚本
端到端学习调度策略
"""

import os
import time
import json
import numpy as np
import torch
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import *
from environment import Environment
from rl_direct_agent import DirectRLAgent
from rl_direct_controller import DirectController, MAX_ACTIONS


class DirectRLTrainer:
    """直接控制RL训练器"""
    
    def __init__(self, 
                 num_episodes: int = 500,
                 use_gpu: bool = False,
                 seed: int = 42):
        """初始化训练器"""
        self.num_episodes = num_episodes
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        
        # 随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 创建环境和控制器
        self.env = Environment(seed=seed)
        self.controller = DirectController(self.env)
        
        # 创建智能体
        state_dim = self.controller.get_state_dim()
        self.agent = DirectRLAgent(
            state_dim=state_dim,
            max_actions=MAX_ACTIONS,
            device=self.device,
            lr=3e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9995,
            buffer_size=100000,
            batch_size=64,
            target_update_freq=100
        )
        
        # 统计
        self.episode_rewards = []
        self.episode_completions = []
        self.episode_losses = []
        self.best_completion = 0
        
        os.makedirs("models", exist_ok=True)
        
        print("直接控制RL训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {state_dim}")
        print(f"  最大动作数: {MAX_ACTIONS}")
        print(f"  回合数: {num_episodes}")
    
    def train_episode(self) -> dict:
        """训练一个episode"""
        obs = self.env.reset()
        state = self.controller.get_state()
        action_list, action_mask = self.controller.build_actions()
        
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        step_count = 0
        decision_count = 0
        
        # 决策计时
        next_decision_time = 0.0
        
        prev_completed = 0
        
        while self.env.current_time < EPISODE_DURATION:
            is_decision_time = (self.env.current_time >= next_decision_time)
            
            high_level_action = None
            
            if is_decision_time:
                # 选择动作
                action_idx = self.agent.select_action(state, action_mask)
                
                # 解码并执行
                high_level_action = self.controller.decode_action(action_idx)
                
                decision_count += 1
                next_decision_time = self.env.current_time + HIGH_LEVEL_DECISION_INTERVAL
            
            # 执行环境step
            prev_completed = self.env.completed_cargos
            next_obs, env_reward, done = self.env.step(high_level_action, low_level_actions=None)
            new_completed = self.env.completed_cargos
            
            # 获取新状态和动作空间
            next_state = self.controller.get_state()
            next_action_list, next_action_mask = self.controller.build_actions()
            
            # 计算即时奖励（仅在决策时刻）
            if is_decision_time:
                reward = self.controller.compute_reward(action_idx, prev_completed, new_completed)
                episode_reward += reward
                
                # 存储经验
                self.agent.store_transition(
                    state, action_idx, reward,
                    next_state, next_action_mask, done
                )
                
                # 训练
                loss = self.agent.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
            
            # 更新状态
            state = next_state
            action_mask = next_action_mask
            step_count += 1
            
            if done:
                break
        
        # 衰减探索率
        self.agent.decay_epsilon()
        
        return {
            'reward': episode_reward,
            'completed': self.env.completed_cargos,
            'steps': step_count,
            'decisions': decision_count,
            'loss': episode_loss / max(1, loss_count),
            'epsilon': self.agent.epsilon,
            'buffer_size': len(self.agent.replay_buffer)
        }
    
    def train(self):
        """训练循环"""
        print("\n" + "=" * 80)
        print("开始直接控制RL训练")
        print("=" * 80)
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            ep_start = time.time()
            stats = self.train_episode()
            ep_time = time.time() - ep_start
            
            self.episode_rewards.append(stats['reward'])
            self.episode_completions.append(stats['completed'])
            self.episode_losses.append(stats['loss'])
            
            print(f"Episode {episode+1:4d}/{self.num_episodes} | "
                  f"奖励: {stats['reward']:8.2f} | "
                  f"完成: {stats['completed']:3d} | "
                  f"决策: {stats['decisions']:5d} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Loss: {stats['loss']:.4f} | "
                  f"缓冲: {stats['buffer_size']:6d} | "
                  f"耗时: {ep_time:.1f}s")
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_completed = np.mean(self.episode_completions[-10:])
                avg_loss = np.mean(self.episode_losses[-10:])
                
                print(f"  [最近10个] 平均奖励: {avg_reward:.2f} | "
                      f"平均完成: {avg_completed:.1f} | "
                      f"平均Loss: {avg_loss:.4f}")
                
                if avg_completed > self.best_completion:
                    self.best_completion = avg_completed
                    self._save_model("best")
                    print(f"  *** 新最佳! 平均完成: {avg_completed:.1f} ***")
                print()
            
            if (episode + 1) % 50 == 0:
                self._save_model(f"checkpoint_ep{episode+1}")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)
        print(f"总耗时: {total_time/60:.2f}分钟")
        print(f"平均完成: {np.mean(self.episode_completions):.1f}")
        print(f"最佳平均完成: {self.best_completion:.1f}")
        
        self._save_model("final")
        self._save_stats()
    
    def _save_model(self, prefix: str):
        path = f"models/direct_rl_{prefix}.pt"
        self.agent.save(path)
        print(f"  模型已保存: {path}")
    
    def _save_stats(self):
        stats = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_completions': [int(x) for x in self.episode_completions],
            'episode_losses': [float(x) for x in self.episode_losses],
            'best_completion': float(self.best_completion)
        }
        path = "models/direct_rl_stats.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"  统计已保存: {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='直接控制RL训练')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available()
    print(f"GPU可用: {use_gpu}")
    if use_gpu:
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()
    
    trainer = DirectRLTrainer(
        num_episodes=args.episodes,
        use_gpu=use_gpu,
        seed=args.seed
    )
    trainer.train()


if __name__ == "__main__":
    main()
