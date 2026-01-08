"""
直接控制RL智能体
使用DQN with Action Masking实现端到端学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Optional
from config import *


class DQNNetwork(nn.Module):
    """DQN网络，支持动作掩码"""
    
    def __init__(self, state_dim: int, max_actions: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions)
        )
    
    def forward(self, x: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, state_dim]
            action_mask: 动作掩码 [batch_size, max_actions], 1=有效, 0=无效
        
        Returns:
            Q值 [batch_size, max_actions]
        """
        q_values = self.net(x)
        
        if action_mask is not None:
            # 将无效动作的Q值设为极小值
            q_values = q_values.masked_fill(action_mask == 0, -1e9)
        
        return q_values


class DirectRLAgent:
    """直接控制RL智能体"""
    
    def __init__(self, 
                 state_dim: int,
                 max_actions: int,
                 device: str = 'cpu',
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        初始化智能体
        
        Args:
            state_dim: 状态维度
            max_actions: 最大动作数（固定大小，用于网络输出）
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.max_actions = max_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 探索率
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 网络
        self.q_network = DQNNetwork(state_dim, max_actions).to(device)
        self.target_network = DQNNetwork(state_dim, max_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 训练计数
        self.train_steps = 0
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        选择动作（ε-贪心，带动作掩码）
        
        Args:
            state: 状态向量
            action_mask: 动作掩码，1=有效，0=无效
        
        Returns:
            选择的动作索引
        """
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:
            return 0  # 无有效动作时返回NOOP
        
        # ε-贪心
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # 贪心选择
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor, mask_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, next_action_mask: np.ndarray, done: bool):
        """存储经验"""
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_action_mask': next_action_mask,
            'done': done
        })
    
    def train(self) -> Optional[float]:
        """训练一个batch"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor(np.array([t['state'] for t in batch])).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in batch])).to(self.device)
        next_masks = torch.FloatTensor(np.array([t['next_action_mask'] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch]).to(self.device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值（使用动作掩码）
        with torch.no_grad():
            next_q = self.target_network(next_states, next_masks).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 损失
        loss = nn.functional.mse_loss(current_q, target_q)
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        
        # 更新目标网络
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.update_count += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_steps = checkpoint.get('train_steps', 0)
