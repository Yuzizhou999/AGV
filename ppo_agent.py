"""
自定义PPO智能体实现
不依赖Stable-Baselines3，直接从经验缓冲区学习
完全适配多智能体环境
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from config import *


class ActorCritic(nn.Module):
    """
    Actor-Critic网络
    Actor输出动作的均值和标准差（连续动作空间）
    Critic输出状态价值
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化网络
        
        Args:
            obs_dim: 观测维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: 输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )
        
        # Actor: 输出动作标准差的对数
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        """前向传播"""
        features = self.feature_extractor(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(features)
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """
        获取动作
        
        Args:
            obs: 观测
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
        else:
            # 从正态分布采样
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        # 计算对数概率
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        """
        评估给定动作的价值和概率
        用于训练时计算策略梯度
        
        Args:
            obs: 观测
            actions: 动作
            
        Returns:
            log_probs: 动作的对数概率
            values: 状态价值
            entropy: 策略熵
        """
        action_mean, action_std, values = self.forward(obs)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values, entropy


class PPOAgent:
    """
    PPO智能体 - 自定义实现
    可以直接从经验缓冲区学习，不需要Gym环境
    """
    
    def __init__(self, obs_dim: int, action_dim: int, device='cpu',
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, n_epochs=10, batch_size=64):
        """
        初始化PPO智能体
        
        Args:
            obs_dim: 观测空间维度
            action_dim: 动作空间维度
            device: 设备 ('cpu' 或 'cuda')
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵损失系数
            max_grad_norm: 梯度裁剪阈值
            n_epochs: 每次更新训练的轮数
            batch_size: 批量大小
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # 创建网络
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲区
        self.reset_buffer()
    
    def reset_buffer(self):
        """重置经验缓冲区"""
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, obs: np.ndarray, deterministic=False):
        """
        选择动作
        
        Args:
            obs: 观测 (numpy array)
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作值 (numpy array)
            value: 状态价值 (float)
            log_prob: 对数概率 (float)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
            
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()
        
        return action, value, log_prob
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """
        存储一个转移到缓冲区
        
        Args:
            obs: 观测
            action: 动作
            reward: 奖励
            value: 状态价值
            log_prob: 对数概率
            done: 是否结束
        """
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value=0.0):
        """
        计算广义优势估计(GAE)
        
        Args:
            next_value: 最后一个状态的价值（如果episode未结束）
            
        Returns:
            advantages: 优势值
            returns: 回报
        """
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        # 添加最后一个状态的价值
        values = np.append(values, next_value)
        
        # 计算GAE
        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[-1]
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]  # 移除添加的next_value
        
        return advantages, returns
    
    def update(self, next_value=0.0):
        """
        使用缓冲区中的经验更新策略
        
        Args:
            next_value: 最后一个状态的价值（如果episode未结束）
            
        Returns:
            stats: 训练统计信息
        """
        # 检查缓冲区是否为空
        if len(self.buffer['obs']) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(next_value)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 训练统计
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # 多轮训练
        for _ in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(obs))
            
            # 分批训练
            for start_idx in range(0, len(obs), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(obs))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取批量数据
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估动作
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze()
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 计算策略损失（PPO-Clip）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 计算总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 记录统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # 清空缓冲区
        self.reset_buffer()
        
        # 返回统计信息
        stats = {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1)
        }
        
        return stats
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
